#конвертация pdf
import os
import subprocess

pdf_dir = 'docs_pdf/УПД_202007230000_НТКА-002288_002'
output_dir = 'images_jpg'

def image_exporter(filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name = filename.replace('.', ' ').split()[0]
    pdf_path = pdf_dir + '/' + filename
    output_path = output_dir + '/' + name

    cmd = ['pdfimages', '-all', pdf_path, output_path]
    subprocess.call(cmd)
    print('Images extracted: ' + filename)
    #print(os.listdir(output_dir))


files_list = os.listdir(pdf_dir)
print('number_files:', len(files_list))
#print(files_list)

for i in range(len(files_list)):
  image_exporter(files_list[i])