#2.1 Įdiekite „zipfile“ paketą (yra „Python“).
#2.2 Sukurkite klasę pavadinimu FileCompressor.
#2.3 Implementuokite metodą compress(files, output_zip), kad suspaustumėte failų sąrašą į ZIP failą.
#2.4 Implementuokite metodą decompress(zip_file, output_dir), kad ištrauktumėte ZIP failo turinį.

import zipfile

class FileCompressor:
    def compress(self, files, output_zip):
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for file in files:
                zipf.write(file)

    def decompress(self, zip_file, output_dir):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)


fc = FileCompressor()
files_to_compress = ['file1.txt', 'file2.txt']
fc.compress(files_to_compress, 'compressed_files.zip')
fc.decompress('compressed_files.zip', 'extracted_files')