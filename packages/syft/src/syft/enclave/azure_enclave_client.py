# future
from __future__ import annotations

# stdlib
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import cast

# third party
from pydantic import validator
import requests

# relative
from ..core.common.serde.deserialize import _deserialize as deserialize
from ..core.common.serde.serializable import serializable
from ..core.common.uid import UID
from ..core.node.new.api import SyftAPI
from ..core.node.new.base import SyftBaseModel
from ..core.node.new.client import HTTPConnection
from ..core.node.new.client import Routes
from ..core.node.new.client import SyftClient
from ..core.node.new.credentials import SyftSigningKey

if TYPE_CHECKING:
    # relative
    from ..core.node.new.user_code import SubmitUserCode


class EnclaveMetadata(SyftBaseModel):
    """Contains metadata to connect to a specific Enclave"""

    pass


class EnclaveClient(SyftBaseModel):
    """Base EnclaveClient class

    Args:
        owners (List[SyftClient]): The list of Domain Owners the enclave belongs to
        signing_key (SyftSigningKey): The siginig key to be use to sign the message of the client
        url (str): Connection URL to communicate with the enclave.
    """

    owners: List[SyftClient]
    signing_key: Optional[SyftSigningKey] = None
    url: str
    syft_api: Optional[SyftAPI] = None
    # Private _ variables are not accessible by pydantic

    @validator("signing_key", always=True)
    def create_signing_key(cls, v):
        if v is None:
            return SyftSigningKey.generate()
        return v

    def _get_api(self) -> SyftAPI:
        req = requests.get(
            self.url + Routes.ROUTE_API.value,
        )
        obj = deserialize(req.content, from_bytes=True)
        # TODO 🟣 Retrieve of signing key of user after permission  is fully integrated
        obj.signing_key = self.signing_key
        obj.connection = HTTPConnection(self.url)
        return cast(SyftAPI, obj)

    def _set_api(self):
        _api = self._get_api()
        # APIRegistry.set_api_for(node_uid=self.id, api=_api)
        setattr(self, "syft_api", _api)

    @property
    def api(self) -> SyftAPI:
        if self.syft_api is None:
            self._set_api()

        return self.syft_api

    def refresh(self) -> None:
        self._set_api()

    def _get_enclave_metadata(self) -> EnclaveMetadata:
        raise NotImplementedError(
            "EnclaveClient subclasses must implement _get_enclave_metadata()"
        )

    def request_code_execution(self, code: SubmitUserCode):
        # relative
        from ..core.node.new.user_code import SubmitUserCode

        if not isinstance(code, SubmitUserCode):
            raise Exception(
                f"The input code should be of type: {SubmitUserCode} got:{type(code)}"
            )

        enclave_metadata = self._get_enclave_metadata()

        code_id = UID()
        code.id = code_id
        code.enclave_metadata = enclave_metadata

        for domain_client in self.owners:
            domain_client.api.services.code.request_code_execution(code=code)

        res = self.api.services.code.request_code_execution(code=code)

        return res


@serializable(recursive_serde=True)
class AzureEnclaveMetadata(EnclaveMetadata):
    url: str


class AzureEnclaveClient(EnclaveClient):
    def _get_enclave_metadata(self) -> AzureEnclaveMetadata:
        return AzureEnclaveMetadata(url=self.url)

    @staticmethod
    def from_enclave_metadata(
        enclave_metadata: AzureEnclaveMetadata, signing_key: SyftSigningKey
    ) -> AzureEnclaveClient:
        # In the context of Domain Owners, who would like to only communicate with enclave, would not provide owners
        return AzureEnclaveClient(
            owners=[], url=enclave_metadata.url, signing_key=signing_key
        )
