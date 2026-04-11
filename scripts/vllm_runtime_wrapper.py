import sys
import unittest.mock

# --- CAGE HOTFIX: Bypass vLLM Mac circular dependency bugs in memory ---
# vLLM routinely crashes trying to globally import qwen2_5_omni config 
# on older transformers, but crashes natively on newer transformers. 
# We mock the bleeding-edge architecture module only in-memory so vLLM 
# successfully parses the rest of its startup registration hooks natively.
sys.modules['transformers.models.qwen2_5_omni'] = unittest.mock.MagicMock()
sys.modules['transformers.models.qwen2_5_omni.configuration_qwen2_5_omni'] = unittest.mock.MagicMock()

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main
    # We replaced 'vllm serve' with our wrapper, so we append the 'serve' command.
    sys.argv.insert(1, 'serve')
    main()
