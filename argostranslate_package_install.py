import os
from dotenv import load_dotenv
import argostranslate.package
import argostranslate.translate

load_dotenv()
translate_from_code = os.environ.get("TRANSLATE_FROM_CODE")
translate_to_code = os.environ.get("TRANSLATE_TO_CODE")
codes = [[translate_from_code, translate_to_code], [translate_to_code, translate_from_code]]

argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

for cd in codes:
    print(cd)
    package_to_install = next(
        filter(
            lambda x: x.from_code == cd[0] and x.to_code == cd[1], available_packages
        )
    )
    print(package_to_install)
    argostranslate.package.install_from_path(package_to_install.download())

str = "The packages of languages to be translated was installed."
str = argostranslate.translate.translate(str, translate_to_code, translate_from_code)
print(str)
str = argostranslate.translate.translate(str, translate_from_code, translate_to_code)
print(str)

file = "source_documents/state_of_the_union.txt"
if not os.path.exists(file):
    f = open(file.replace(".txt",".en"), "r", encoding="utf-8")
    str = f.read()
    f.close()
    print(f"Translating {file} ...")
    str = argostranslate.translate.translate(str, translate_to_code, translate_from_code)
    f = open(file, "w", encoding="utf-8")
    f.write(str)
    f.close()
