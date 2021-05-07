from hybridrl.agents.ddpg import DDPG
from hybridrl.agents.rs_mpc import RSMPC
from hybridrl.agents.cem import CEM
from hybridrl.agents.rs_mpc_ddpg import RSMPCDDPG


AGENTS = {
    'DDPG': DDPG,
    'RSMPC': RSMPC,
    'CEM': CEM,
    'RSMPCDDPG': RSMPCDDPG
}
