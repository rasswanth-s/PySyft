# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from pydantic import Field
from pydantic import validator
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..types.uid import UID
from ..util.util import thread_ident
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .kv_document_store import KeyValueBackingStore
from .kv_document_store import KeyValueStorePartition
from .locks import FileLockingConfig
from .locks import LockingConfig
from .locks import SyftLock

# here we can create a single connection per cache_key
# since pytest is concurrent processes, we need to isolate each connection
# by its filename and optionally the thread that its running in
# we keep track of each SQLiteBackingStore init in REF_COUNTS
# when it hits 0 we can close the connection and release the file descriptor
SQLITE_CONNECTION_POOL_DB: Dict[str, sqlite3.Connection] = {}
SQLITE_CONNECTION_POOL_CUR: Dict[str, sqlite3.Cursor] = {}
REF_COUNTS: Dict[str, int] = defaultdict(int)


def cache_key(db_name: str) -> str:
    return f"{db_name}_{thread_ident()}"


def _repr_debug_(value: Any) -> str:
    if hasattr(value, "_repr_debug_"):
        return str(value._repr_debug_())
    return repr(value)


def raise_exception(table_name: str, e: Exception):
    if "disk I/O error" in str(e):
        message = f"Error usually related to concurrent writes. {str(e)}"
        raise Exception(message)

    if "Cannot operate on a closed database" in str(e):
        message = (
            "Error usually related to calling self.db.close()"
            + f"before last SQLiteBackingStore.__del__ gets called. {str(e)}"
        )
        raise Exception(message)

    # if its something else other than "table already exists" raise original e
    if f"table {table_name} already exists" not in str(e):
        raise e


@serializable(attrs=["index_name", "settings", "store_config"])
class SQLiteBackingStore(KeyValueBackingStore):
    """Core Store logic for the SQLite stores.

    Parameters:
        `index_name`: str
            Index name
        `settings`: PartitionSettings
            Syft specific settings
        `store_config`: SQLiteStoreConfig
            Connection Configuration
        `ddtype`: Type
            Class used as fallback on `get` errors
    """

    def __init__(
        self,
        index_name: str,
        settings: PartitionSettings,
        store_config: StoreConfig,
        ddtype: Optional[type] = None,
    ) -> None:
        self.index_name = index_name
        self.settings = settings
        self.store_config = store_config
        self._ddtype = ddtype
        self.file_path = self.store_config.client_config.file_path
        self.db_filename = store_config.client_config.filename

        # if tempfile.TemporaryDirectory() varies from process to process
        # could this cause different locks on the same file
        temp_dir = tempfile.TemporaryDirectory().name
        lock_path = Path(temp_dir) / "sqlite_locks" / self.db_filename
        self.lock_config = FileLockingConfig(client_path=lock_path)
        self.create_table()
        REF_COUNTS[cache_key(self.db_filename)] += 1

    @property
    def table_name(self) -> str:
        return f"{self.settings.name}_{self.index_name}"

    def _connect(self) -> None:
        # SQLite is not thread safe by default so we ensure that each connection
        # comes from a different thread. In cases of Uvicorn and other AWSGI servers
        # there will be many threads handling incoming requests so we need to ensure
        # that different connections are used in each thread. By using a dict for the
        # _db and _cur we can ensure they are never shared

        path = Path(self.file_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(
            self.file_path,
            timeout=self.store_config.client_config.timeout,
            check_same_thread=False,  # do we need this if we use the lock?
            # check_same_thread=self.store_config.client_config.check_same_thread,
        )
        # TODO: Review OSX compatibility.
        # Set journal mode to WAL.
        # connection.execute("pragma journal_mode=wal")
        SQLITE_CONNECTION_POOL_DB[cache_key(self.db_filename)] = connection

    def create_table(self) -> None:
        try:
            with SyftLock(self.lock_config):
                self.cur.execute(
                    f"create table {self.table_name} (uid VARCHAR(32) NOT NULL PRIMARY KEY, "  # nosec
                    + "repr TEXT NOT NULL, value BLOB NOT NULL, "  # nosec
                    + "sqltime TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL)"  # nosec
                )
                self.db.commit()
        except Exception as e:
            raise_exception(self.table_name, e)

    @property
    def db(self) -> sqlite3.Connection:
        if cache_key(self.db_filename) not in SQLITE_CONNECTION_POOL_DB:
            self._connect()
        return SQLITE_CONNECTION_POOL_DB[cache_key(self.db_filename)]

    @property
    def cur(self) -> sqlite3.Cursor:
        if cache_key(self.db_filename) not in SQLITE_CONNECTION_POOL_CUR:
            SQLITE_CONNECTION_POOL_CUR[cache_key(self.db_filename)] = self.db.cursor()

        return SQLITE_CONNECTION_POOL_CUR[cache_key(self.db_filename)]

    def _close(self) -> None:
        self._commit()
        REF_COUNTS[cache_key(self.db_filename)] -= 1
        if REF_COUNTS[cache_key(self.db_filename)] <= 0:
            # once you close it seems like other object references can't re-use the
            # same connection
            self.db.close()
            del SQLITE_CONNECTION_POOL_DB[cache_key(self.db_filename)]
        else:
            # don't close yet because another SQLiteBackingStore is probably still open
            pass

    def _commit(self) -> None:
        self.db.commit()

    def _execute(
        self, sql: str, *args: Optional[List[Any]]
    ) -> Result[Ok[sqlite3.Cursor], Err[str]]:
        with SyftLock(self.lock_config):
            cursor: Optional[sqlite3.Cursor] = None
            err = None
            try:
                cursor = self.cur.execute(sql, *args)
            except Exception as e:
                raise_exception(self.table_name, e)

            # TODO: Which exception is safe to rollback on?
            # we should map out some more clear exceptions that can be returned
            # rather than halting the program like disk I/O error etc
            # self.db.rollback()  # Roll back all changes if an exception occurs.
            # err = Err(str(e))
            self.db.commit()  # Commit if everything went ok

            if err is not None:
                return err

            return Ok(cursor)

    def _set(self, key: UID, value: Any) -> None:
        if self._exists(key):
            self._update(key, value)
        else:
            insert_sql = (
                f"insert into {self.table_name} (uid, repr, value) VALUES (?, ?, ?)"  # nosec
            )
            data = _serialize(value, to_bytes=True)
            res = self._execute(insert_sql, [str(key), _repr_debug_(value), data])
            if res.is_err():
                raise ValueError(res.err())

    def _update(self, key: UID, value: Any) -> None:
        insert_sql = (
            f"update {self.table_name} set uid = ?, repr = ?, value = ? where uid = ?"  # nosec
        )
        data = _serialize(value, to_bytes=True)
        res = self._execute(insert_sql, [str(key), _repr_debug_(value), data, str(key)])
        if res.is_err():
            raise ValueError(res.err())

    def _get(self, key: UID) -> Any:
        select_sql = f"select * from {self.table_name} where uid = ? order by sqltime"  # nosec
        res = self._execute(select_sql, [str(key)])
        if res.is_err():
            raise KeyError(f"Query {select_sql} failed")
        cursor = res.ok()

        row = cursor.fetchone()
        if row is None or len(row) == 0:
            raise KeyError(f"{key} not in {type(self)}")
        data = row[2]
        return _deserialize(data, from_bytes=True)

    def _exists(self, key: UID) -> bool:
        select_sql = f"select uid from {self.table_name} where uid = ?"  # nosec

        res = self._execute(select_sql, [str(key)])
        if res.is_err():
            return False
        cursor = res.ok()

        row = cursor.fetchone()
        if row is None:
            return False

        return bool(row)

    def _get_all(self) -> Any:
        select_sql = f"select * from {self.table_name} order by sqltime"  # nosec
        keys = []
        data = []

        res = self._execute(select_sql)
        if res.is_err():
            return {}
        cursor = res.ok()

        rows = cursor.fetchall()
        if rows is None:
            return {}

        for row in rows:
            keys.append(UID(row[0]))
            data.append(_deserialize(row[2], from_bytes=True))
        return dict(zip(keys, data))

    def _get_all_keys(self) -> Any:
        select_sql = f"select uid from {self.table_name} order by sqltime"  # nosec
        keys = []

        res = self._execute(select_sql)
        if res.is_err():
            return []
        cursor = res.ok()

        rows = cursor.fetchall()
        if rows is None:
            return []

        for row in rows:
            keys.append(UID(row[0]))
        return keys

    def _delete(self, key: UID) -> None:
        select_sql = f"delete from {self.table_name} where uid = ?"  # nosec
        res = self._execute(select_sql, [str(key)])
        if res.is_err():
            raise ValueError(res.err())

    def _delete_all(self) -> None:
        select_sql = f"delete from {self.table_name}"  # nosec
        res = self._execute(select_sql)
        if res.is_err():
            raise ValueError(res.err())

    def _len(self) -> int:
        select_sql = f"select count(uid) from {self.table_name}"  # nosec
        res = self._execute(select_sql)
        if res.is_err():
            raise ValueError(res.err())
        cursor = res.ok()
        cnt = cursor.fetchone()[0]
        return cnt

    def __setitem__(self, key: Any, value: Any) -> None:
        self._set(key, value)

    def __getitem__(self, key: Any) -> Self:
        try:
            return self._get(key)
        except KeyError as e:
            if self._ddtype is not None:
                return self._ddtype()
            raise e

    def __repr__(self) -> str:
        return repr(self._get_all())

    def __len__(self) -> int:
        return self._len()

    def __delitem__(self, key: str):
        self._delete(key)

    def clear(self) -> Self:
        self._delete_all()

    def copy(self) -> Self:
        return deepcopy(self)

    def keys(self) -> Any:
        return self._get_all_keys()

    def values(self) -> Any:
        return self._get_all().values()

    def items(self) -> Any:
        return self._get_all().items()

    def pop(self, key: Any) -> Self:
        value = self._get(key)
        self._delete(key)
        return value

    def __contains__(self, key: Any) -> bool:
        return self._exists(key)

    def __iter__(self) -> Any:
        return iter(self.keys())

    def __del__(self):
        try:
            self._close()
        except BaseException:
            print("Could not close connection")
            pass


@serializable()
class SQLiteStorePartition(KeyValueStorePartition):
    """SQLite StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for indexing and partitioning
        `store_config`: SQLiteStoreConfig
            SQLite specific configuration
    """

    def close(self) -> None:
        self.lock.acquire()
        try:
            # I think we don't want these now, because of the REF_COUNT?
            # self.data._close()
            # self.unique_keys._close()
            # self.searchable_keys._close()
            pass
        except BaseException:
            pass
        self.lock.release()

    def commit(self) -> None:
        self.lock.acquire()
        try:
            self.data._commit()
            self.unique_keys._commit()
            self.searchable_keys._commit()
        except BaseException:
            pass
        self.lock.release()


# the base document store is already a dict but we can change it later
@serializable()
class SQLiteDocumentStore(DocumentStore):
    """SQLite Document Store

    Parameters:
        `store_config`: StoreConfig
            SQLite specific configuration, including connection details and client class type.
    """

    partition_type = SQLiteStorePartition


@serializable()
class SQLiteStoreClientConfig(StoreClientConfig):
    """SQLite connection config

    Parameters:
        `filename` : str
            Database name
        `path` : Path or str
            Database folder
        `check_same_thread`: bool
            If True (default), ProgrammingError will be raised if the database connection is used
            by a thread other than the one that created it. If False, the connection may be accessed
            in multiple threads; write operations may need to be serialized by the user to avoid
            data corruption.
        `timeout`: int
            How many seconds the connection should wait before raising an exception, if the database
            is locked by another connection. If another connection opens a transaction to modify the
            database, it will be locked until that transaction is committed. Default five seconds.
    """

    filename: Optional[str] = None
    path: Union[str, Path] = Field(default_factory=tempfile.gettempdir)
    check_same_thread: bool = True
    timeout: int = 5

    # We need this in addition to Field(default_factory=...)
    # so users can still do SQLiteStoreClientConfig(path=None)
    @validator("path", pre=True)
    def __default_path(cls, path: Optional[Union[str, Path]]) -> Union[str, Path]:
        if path is None:
            return tempfile.gettempdir()
        return path

    @property
    def file_path(self) -> Optional[Path]:
        return Path(self.path) / self.filename if self.filename is not None else None


@serializable()
class SQLiteStoreConfig(StoreConfig):
    __canonical_name__ = "SQLiteStoreConfig"
    """SQLite Store config, used by SQLiteStorePartition

    Parameters:
        `client_config`: SQLiteStoreClientConfig
            SQLite connection configuration
        `store_type`: DocumentStore
            Class interacting with QueueStash. Default: SQLiteDocumentStore
        `backing_store`: KeyValueBackingStore
            The Store core logic. Default: SQLiteBackingStore
        locking_config: LockingConfig
            The config used for store locking. Available options:
                * NoLockingConfig: no locking, ideal for single-thread stores.
                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.
                * FileLockingConfig: file based locking, ideal for same-device different-processes/threads stores.
                * RedisLockingConfig: Redis-based locking, ideal for multi-device stores.
            Defaults to FileLockingConfig.
    """

    client_config: SQLiteStoreClientConfig
    store_type: Type[DocumentStore] = SQLiteDocumentStore
    backing_store: Type[KeyValueBackingStore] = SQLiteBackingStore
    locking_config: LockingConfig = FileLockingConfig()
