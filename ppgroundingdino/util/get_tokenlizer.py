import paddle
from paddlenlp.transformers import AutoTokenizer, BertModel, RobertaModel
from paddlenlp.taskflow.utils import pad_batch_data

def process_caption(tokenizer, captions, max_seq_len):
    tokenized = {}
    input_ids = tokenizer(captions).input_ids
    input_ids = pad_batch_data(input_ids)
    input_ids = paddle.to_tensor(input_ids, dtype = paddle.int64).squeeze(-1)
    tokenized['input_ids'] = input_ids
    tokenized['attention_mask'] = paddle.cast(input_ids != 0, paddle.int64)
    return tokenized


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
