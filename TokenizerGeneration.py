from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

# Read the text file and extract unique words as custom tokenizer.
unique_words = set()
with open("/home/ubuntu/Documents/TokyoPT/PTChain/SumPTChainAttrHalf.txt", "r") as file:
    for line in file:
        words = line.strip().split()
        unique_words.update(words)

all_tokens = list(unique_words) + ["[UNK]", "[PAD]", "[EOS]"]
special_tokens = ["[UNK]", "[PAD]", "[EOS]"]
vocab = {token: i for i, token in enumerate(all_tokens)}

# Create a tokenizer with this vocabulary
model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
tokenizer = Tokenizer(model)
tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.WordPiece()
tokenizer.add_special_tokens(special_tokens)
tokenizer.save("/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer_attr.json")
print("Total number of tokens in the tokenizer:", len(all_tokens))


# Test tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer_attr.json")
tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[EOS]"

encoded = tokenizer.encode("House Store_Daily Back_Home Pickup_Drop_Off 15 [EOS] Male")
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)
