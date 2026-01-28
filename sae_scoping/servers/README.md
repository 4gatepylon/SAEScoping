This provides a server and client system to more easily test models.
- `hf_openai_server_gemma_one_model_only.py` is meant for only Gemma2 with SAELens SAEs if desired.
- `hf_openai_server.py` is meant for Gemma2 or Llama2 and should support future models too so long as you use Sparsify, SAELens, or no SAEs.
- `hf_openai_cli_client.py` should work with any server

Importantly, these do batched inference by accumulating requests in a queue. That makes it in practice 16x or more faster.