# stdlib
from typing import List
from typing import Optional
from typing import cast

# third party
from pydantic import validator
import requests

# relative
from ..core.common.serde.deserialize import _deserialize as deserialize
from ..core.node.new.api import SyftAPI
from ..core.node.new.base import SyftBaseModel
from ..core.node.new.client import HTTPConnection
from ..core.node.new.client import Routes
from ..core.node.new.client import SyftClient
from ..core.node.new.credentials import SyftSigningKey


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

    @validator("signing_key")
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


class AzureEnclaveClient(EnclaveClient):
    pass
