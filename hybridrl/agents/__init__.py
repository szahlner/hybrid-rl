from hybridrl.agents.ddpg import DDPG
from hybridrl.agents.rs_mpc import RSMPC
from hybridrl.agents.cem import CEM


AGENTS = {
    'DDPG': DDPG,
    'RSMPC': RSMPC,
    'CEM': CEM
}
