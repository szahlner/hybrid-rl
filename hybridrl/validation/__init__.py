from hybridrl.validation.experiment_schema import EXPERIMENT_SCHEMA
from hybridrl.validation.environment_schema import ENVIRONMENT_SCHEMA
from hybridrl.validation.test_schema import TEST_SCHEMA
from hybridrl.validation.ddpg_schema import DDPG_SCHEMA
from hybridrl.validation.rs_mpc_schema import RSMPC_SCHEMA
from hybridrl.validation.cem_schema import CEM_SCHEMA
from hybridrl.validation.rs_mpc_ddpg_schema import RSMPC_DDPG_SCHEMA


AGENT_SCHEMAS = {
    'DDPG': DDPG_SCHEMA,
    'RSMPC': RSMPC_SCHEMA,
    'CEM': CEM_SCHEMA,
    'RSMPCDDPG': RSMPC_DDPG_SCHEMA
}
