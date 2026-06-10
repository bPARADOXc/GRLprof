import torch
from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

#在文本进行 Tokenization 时，正确地保留和插入特殊的 <graph> 占位符 Token，防止它被 Tokenizer 拆碎或丢失。
def tokenizer_graph_token(
    prompt,
    tokenizer,
    graph_token_index=GRAPH_TOKEN_INDEX,
    return_tensors=None,
    add_special_tokens=True,
):
    #首先按 <graph> (即 DEFAULT_GRAPH_TOKEN) 把整个字符串切开
    # 对切开后的每一段纯文本分别进行 Tokenization。
    #当直接调用 tokenizer(chunk) 而不指定 padding 参数时，HuggingFace Tokenizer 的默认行为是不进行任何 Padding的。它只会返回该文本实际对应的 Token ID 列表。
    prompt_chunks = [
        tokenizer(chunk, verbose=False, add_special_tokens=add_special_tokens).input_ids
        #？？？
        for chunk in prompt.split(DEFAULT_GRAPH_TOKEN)
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        add_special_tokens
        and len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [graph_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_length, int) and max_length > 0 and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
