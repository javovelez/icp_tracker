

f1 = open('F:/Documentos/Dharma/captura_2_2023/capturas/verticales/input.txt', 'r')
f2 = open('F:/Documentos/Dharma/captura_2_2023/tracked/auxil.txt', 'r')
f3 = open('F:/Documentos/Dharma/captura_2_2023/tracked/input.txt', 'w')

source = f1.readlines()
reference = f2.readlines()

f1.close()
f2.close()

reference = [r.replace('\n', '.mp4') for r in reference]
source = [s.replace('\n', '') for s in source]
print(source)
for s in source:
    for r in reference:
        if s.endswith(r):
            print(s)
            f3.write(s+'\n')
f3.close()