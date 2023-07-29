from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

tokens = ["Commute", "House", "Office", "Store_Daily", "Go_School", "School", "Back_Home",
          "Shopping_Daily", "Shopping_Nondaily", "Store_Nondaily", "Go_Eat", "Socializing",
          "Go_Recreational_Facility", "Pickup_Drop_Off", "Go_Sightseeing", "Tourist_Spot",
          "Private_Movement", "Private_Space", "Delivering", "Business_Place", "Attend_Meeting",
          "Go_Occupation", "Go_Agricultural_Work", "Natural_Area", "Go_Other_Business",
          "Go_Exercise", "Pitch", "Volunteering", "Public_Space", "Welcoming"]
special_tokens = ["[UNK]", "[PAD]", "[EOS]"]
all_tokens = special_tokens + tokens

vocab = {token: i for i, token in enumerate(all_tokens)}
model = models.WordLevel(vocab=vocab, unk_token="[UNK]")

tokenizer = Tokenizer(model)
tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.WordPiece()
tokenizer.add_special_tokens(special_tokens)
tokenizer.save("/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")

# Test tokenizer
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/ubuntu/Documents/Tokenizer/trip_chain_tokenizer.json")
# tokenizer.pad_token = "[PAD]"
# tokenizer.eos_token = "[EOS]"
#
# encoded = tokenizer.encode("Commute Store_Daily Back_Home Pickup_Drop_Off [EOS]")
# print(encoded)
#
# decoded = tokenizer.decode(encoded)
# print(decoded)

