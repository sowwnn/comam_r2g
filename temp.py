from modules.tokenizers import Tokenizer
from config.configs import Test_Config


args = Test_Config()
tokenizer = Tokenizer(args)

for item in ['effusion', 'pleural', 'cardiomediastinal']:
    print(tokenizer.get_id_by_token(item))
