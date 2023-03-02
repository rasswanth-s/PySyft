# stdlib
from typing import Dict
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from .....enclave.azure_enclave_client import AzureEnclaveClient
from .....enclave.azure_enclave_client import AzureEnclaveMetadata
from .....oblv.deployment_client import OblvMetadata
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ..api import UserNodeView
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..document_store import DocumentStore
from ..service import AbstractService
from ..service import service_method
from ..task.oblv_service import DictObject
from ..task.oblv_service import OblvService
from ..user_code import UserCode
from ..user_code import UserCodeStatus


# TODO 🟣 Created a generic Enclave Service
@serializable(recursive_serde=True)
class AzureEnclaveService(AbstractService):
    store: DocumentStore
    service_name: Optional[str]

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    @service_method(
        path="enclave.send_user_code_inputs_to_enclave",
        name="send_user_code_inputs_to_enclave",
    )
    def send_user_code_inputs_to_enclave(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        inputs: Dict,
        node_name: str,
    ) -> Result[Ok, Err]:
        user_code_service = context.node.get_service("usercodeservice")
        action_service = context.node.get_service("actionservice")
        user_code = user_code_service.get_by_uid(context, uid=user_code_id)

        res = user_code.status.mutate(
            value=UserCodeStatus.EXECUTE,
            node_name=node_name,
            verify_key=context.credentials,
        )

        if res.is_err():
            return res
        user_code.status = res.ok()

        user_code_service.update_code_state(context=context, code_item=user_code)

        if not action_service.exists(context=context, obj_id=user_code_id):
            dict_object = DictObject(id=user_code_id)
            dict_object.base_dict[str(context.credentials)] = inputs
            action_service.store.set(
                uid=user_code_id,
                credentials=context.node.signing_key.verify_key,
                syft_object=dict_object,
            )

        else:
            res = action_service.store.get(
                uid=user_code_id, credentials=context.node.signing_key.verify_key
            )
            if res.is_ok():
                dict_object = res.ok()
                dict_object.base_dict[str(context.credentials)] = inputs
                action_service.store.set(
                    uid=user_code_id,
                    credentials=context.node.signing_key.verify_key,
                    syft_object=dict_object,
                )
            else:
                return res

        return Ok(Ok(True))


# Checks if the given user code would  propogate value to enclave on acceptance
def check_enclave_transfer(
    user_code: UserCode, value: UserCodeStatus, context: ChangeContext
):
    if (
        isinstance(user_code.enclave_metadata, OblvMetadata)
        and value == UserCodeStatus.EXECUTE
    ):
        method = context.node.get_service_method(OblvService.get_api_for)

        api = method(
            user_code.enclave_metadata,
            context.node.signing_key,
        )
        # send data of the current node to enclave
        user_node_view = UserNodeView(
            node_name=context.node.name, verify_key=context.node.signing_key.verify_key
        )
        inputs = user_code.input_policy.inputs[user_node_view]
        action_service = context.node.get_service("actionservice")
        for var_name, uid in inputs.items():
            action_object = action_service.store.get(
                uid=uid, credentials=context.node.signing_key.verify_key
            )
            if action_object.is_err():
                return action_object
            inputs[var_name] = action_object.ok()

        res = api.services.oblv.send_user_code_inputs_to_enclave(
            user_code_id=user_code.id, inputs=inputs, node_name=context.node.name
        )

        return res

    elif (
        isinstance(user_code.enclave_metadata, AzureEnclaveMetadata)
        and value == UserCodeStatus.EXECUTE
    ):
        # TODO 🟣 Restructure url it work for local mode host.docker.internal
        azure_enclave_client = AzureEnclaveClient.from_enclave_metadata(
            enclave_metadata=user_code.enclave_metadata,
            signing_key=context.node.signing_key,
        )

        # send data of the current node to enclave
        user_node_view = UserNodeView(
            node_name=context.node.name, verify_key=context.node.signing_key.verify_key
        )
        inputs = user_code.input_policy.inputs[user_node_view]
        action_service = context.node.get_service("actionservice")
        for var_name, uid in inputs.items():
            action_object = action_service.store.get(
                uid=uid, credentials=context.node.signing_key.verify_key
            )
            if action_object.is_err():
                return action_object
            inputs[var_name] = action_object.ok()

        res = (
            azure_enclave_client.api.services.enclave.send_user_code_inputs_to_enclave(
                user_code_id=user_code.id, inputs=inputs, node_name=context.node.name
            )
        )

        return res
    else:
        return Ok()
