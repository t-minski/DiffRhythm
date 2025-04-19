# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from .mandarin import chinese_to_ipa
from .english import english_to_ipa
from .french import french_to_ipa
from .korean import korean_to_ipa
from .german import german_to_ipa


def cjekfd_cleaners(text, sentence, language, text_tokenizers):

    if language == "zh":
        return chinese_to_ipa(text, sentence, text_tokenizers["zh"])
    elif language == "en":
        return english_to_ipa(text, text_tokenizers["en"])
    elif language == "fr":
        return french_to_ipa(text, text_tokenizers["fr"])
    elif language == "ko":
        return korean_to_ipa(text, text_tokenizers["ko"])
    elif language == "de":
        return german_to_ipa(text, text_tokenizers["de"])
    else:
        raise Exception("Unknown language: %s" % language)
        return None
