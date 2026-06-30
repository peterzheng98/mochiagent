from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.models.gpt import GPTModel
import torch

def apply_all_inputs_embeds_patches():
    """
    Patch all related classes to add inputs_embeds support
    """
    
    # First patch embedding class
    def patch_embeddings():
        # GPT2Embeddings patch
        original_gpt2_forward = GPT2Embeddings.forward
        def gpt2_forward_with_inputs_embeds(self, input_ids=None, position_ids=None, inputs_embeds=None):
            if inputs_embeds is not None:
                batch_size, seqlen = inputs_embeds.shape[:2]
                embeddings = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds must be provided")
                return original_gpt2_forward(self, input_ids, position_ids)
            
            if self.max_position_embeddings > 0:
                if position_ids is None:
                    position_ids = torch.arange(seqlen, dtype=torch.long, device=embeddings.device)
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings
                
            return embeddings
        
        GPT2Embeddings.forward = gpt2_forward_with_inputs_embeds
        
        # ParallelGPT2Embeddings patch
        original_parallel_forward = ParallelGPT2Embeddings.forward
        def parallel_forward_with_inputs_embeds(self, input_ids=None, position_ids=None, inputs_embeds=None, combine_batch_seqlen_dim=False):
            if inputs_embeds is not None:
                batch_size, seqlen = inputs_embeds.shape[:2]
                embeddings = inputs_embeds
                
                world_size = torch.distributed.get_world_size(self.process_group)
                if self.max_position_embeddings > 0:
                    if position_ids is None:
                        position_ids = torch.arange(seqlen, dtype=torch.long, device=embeddings.device)
                    position_embeddings = self.position_embeddings(position_ids)
                    
                    if world_size <= 1:
                        embeddings = embeddings + position_embeddings
                    else:
                        partition_dim = self.position_embeddings.embedding_dim
                        rank = torch.distributed.get_rank(self.process_group)
                        embeddings[
                            ..., rank * partition_dim : (rank + 1) * partition_dim
                        ] += position_embeddings
                
                if combine_batch_seqlen_dim:
                    from einops import rearrange
                    embeddings = rearrange(embeddings, "b s d -> (b s) d")
                
                if world_size <= 1:
                    return embeddings
                else:
                    from flash_attn.utils.distributed import reduce_scatter, all_reduce
                    reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                    return reduce_fn(embeddings, self.process_group)
            else:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds must be provided")
                return original_parallel_forward(self, input_ids, position_ids, combine_batch_seqlen_dim)
        
        ParallelGPT2Embeddings.forward = parallel_forward_with_inputs_embeds
    
    # Then patch GPTModel
    def patch_gpt_model():
        original_forward = GPTModel.forward
        
        def new_forward(self, input_ids=None, position_ids=None, inputs_embeds=None, inference_params=None):
            if inputs_embeds is not None:
                if input_ids is not None:
                    raise ValueError("Cannot provide both input_ids and inputs_embeds")
                batch_size, seqlen = inputs_embeds.shape[:2]
                
                # Handle embedding
                embedding_kwargs = (
                    {"combine_batch_seqlen_dim": True}
                    if self.process_group is not None and self.sequence_parallel
                    else {}
                )
                
                # Call patched embeddings
                hidden_states = self.embeddings(
                    input_ids=None,
                    position_ids=position_ids, 
                    inputs_embeds=inputs_embeds,
                    **embedding_kwargs
                )
                
                # Apply embeddings_multiplier
                if self.embeddings_multiplier != 1.0:
                    hidden_states = hidden_states * self.embeddings_multiplier
                
                # Set variables for subsequent processing
                if self.parallel_block:
                    hidden_states2 = None
                residual = None
                
                # Handle mixer_kwargs
                mixer_kwargs = (
                    {"seqlen": seqlen}
                    if self.process_group is not None and self.sequence_parallel
                    else {}
                )
                if inference_params is not None:
                    mixer_kwargs["inference_params"] = inference_params
                
                # Pass through layers
                for layer in self.layers:
                    if self.prenorm:
                        if not self.parallel_block:
                            hidden_states, residual = layer(
                                hidden_states, residual, mixer_kwargs=mixer_kwargs
                            )
                        else:
                            hidden_states, hidden_states2, residual = layer(
                                hidden_states, hidden_states2, residual, mixer_kwargs=mixer_kwargs
                            )
                    else:
                        hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                
                # Final processing
                if self.prenorm:
                    if not self.fused_dropout_add_ln:
                        dropped = self.drop_f(hidden_states)
                        if not self.parallel_block:
                            residual = (dropped + residual) if residual is not None else dropped
                        else:
                            dropped2 = self.drop_f(hidden_states2)
                            residual = (
                                (residual + dropped + dropped2)
                                if residual is not None
                                else dropped + dropped2
                            )
                        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
                    else:
                        from flash_attn.ops.layer_norm import layer_norm_fn
                        from flash_attn.ops.rms_norm import RMSNorm
                        hidden_states = layer_norm_fn(
                            hidden_states,
                            self.ln_f.weight,
                            self.ln_f.bias,
                            residual=residual,
                            x1=None if not self.parallel_block else hidden_states2,
                            eps=self.ln_f.eps,
                            dropout_p=self.drop_f.p if self.training else 0.0,
                            prenorm=False,
                            is_rms_norm=isinstance(self.ln_f, RMSNorm)
                        )
                
                return hidden_states
            else:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds must be provided")
                return original_forward(self, input_ids, position_ids, inference_params)
        
        GPTModel.forward = new_forward
    
    # Apply all patches
    patch_embeddings()
    patch_gpt_model()
    
    # print("✅ Successfully patched all classes:")
    # print("  - GPT2Embeddings")
    # print("  - ParallelGPT2Embeddings") 
    # print("  - GPTModel")
    # print("All classes now support inputs_embeds parameter")

# Auto-apply patches
apply_all_inputs_embeds_patches()

    # Re-export patched classes
__all__ = ['GPTModel', 'GPT2Embeddings', 'ParallelGPT2Embeddings']