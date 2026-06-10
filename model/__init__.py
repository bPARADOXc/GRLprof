from .language_model.llaga_llama import LlagaLlamaForCausalLM, LlagaConfig
from .language_model.llaga_opt import LlagaOPTForCausalLM, LlagaOPTConfig

try:
    from .language_model.llaga_mpt import LlagaMPTForCausalLM, LlagaMPTConfig
    LLAGA_MPT_IMPORT_ERROR = None
except Exception as exc:
    LlagaMPTForCausalLM = None
    LlagaMPTConfig = None
    LLAGA_MPT_IMPORT_ERROR = exc
