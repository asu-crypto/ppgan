import numpy as np

file_path = 'log_sec_3_celeba.txt'
fr = open(file_path, "r")
lines = fr.read().splitlines()

psnrs = []
ssims = []
fmses = []

for line in lines:
    if "PSNR" in line:
        psnr = line.split()[-1][:-1]
        psnr = float(psnr)
        psnrs.append(psnr)
    elif "SSIM" in line:
        ssim = float(line.split()[-1][:-1])
        ssims.append(ssim)
    elif "FMSE" in line:
        fmse = line.split()[-1][:-1]
        fmse = float(fmse)
        fmses.append(fmse)

fr.close()
for i in range(8):
    avg_psnr, min_psnr, max_psnr = sum(psnrs[i*5:(i+1)*5])/5,min(psnrs[i*5:(i+1)*5]),max(psnrs[i*5:(i+1)*5])
    avg_ssim, min_ssim, max_ssim = sum(ssims[i*5:(i+1)*5])/5,min(ssims[i*5:(i+1)*5]),max(ssims[i*5:(i+1)*5])
    avg_fmse, min_fmse, max_fmse = sum(fmses[i*5:(i+1)*5])/5,min(fmses[i*5:(i+1)*5]),max(fmses[i*5:(i+1)*5])
    print(f"{2**(10-i)}:")
    print(f"{avg_ssim}\t{avg_psnr}\t{avg_fmse}")
    print(f"{max_ssim}\t{max_psnr}\t{max_fmse}")
    print(f"{min_ssim}\t{min_psnr}\t{max_psnr}")