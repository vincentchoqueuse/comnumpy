import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, LogFormatterMathtext

### S3 ###
# df1_1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_S3_1.csv")
# df2_1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S3_1.csv")
# df3_1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S3_1.csv")
# df4_1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S3_1.csv")

# df1_2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_S3_2.csv")
# df2_2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S3_2.csv")
# df3_2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S3_2.csv")
# df4_2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S3_2.csv")

# df1_3 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_S3_3.csv")
# df2_3 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S3_3.csv")
# df3_3 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S3_3.csv")
# df4_3 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S3_3.csv")

# df1_4 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_S3_4.csv")
# df2_4 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S3_4.csv")
# df3_4 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S3_4.csv")
# df4_4 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S3_4.csv")

# df5 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S3_V2.csv")


# # Row-wise mean between A and B
# mean_df1 = (df1_1["SER"] + df1_2["SER"] + df1_3["SER"] + df1_4["SER"]) / 4
# mean_df2 = (df2_1["SER"] + df2_2["SER"] + df2_3["SER"] + df2_4["SER"]) / 4
# mean_df3 = (df3_1["SER"] + df3_2["SER"] + df3_3["SER"] + df3_4["SER"]) / 4
# mean_df4 = (df4_1["SER"] + df4_2["SER"] + df4_3["SER"] + df4_4["SER"]) / 4

### S2 ###
# df1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_S2.csv")
# df2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S2.csv")
# df3 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S2_V2.csv")
# df4 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S2.csv")


df1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_S3_1.csv")
df2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S3_1.csv")
df3 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S3_1.csv")
df4 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S3_1.csv")

plt.figure(figsize=(7, 5))
plt.loglog(df1['dp_tot_T'], df1["SER"], marker='o', linewidth=2, label='MCMA')
plt.loglog(df2['dp_tot_T'], df2["SER"], marker='o', linewidth=2, label='MCMA to Czegledi')
plt.loglog(df3['dp_tot_T'], df3["SER"], marker='o', linewidth=2, label=f'MCMA Soft')
#plt.loglog(df5['dp_tot_T'], df5["SER"], marker='o', linewidth=2, label=f'MCMA Soft alpha=0.5')
plt.loglog(df4['dp_tot_T'], df4["SER"], marker='o', linewidth=2, label=f'DD-Czegledi')
plt.xlabel(r'$\Delta p_{\mathrm{tot}} \cdot T$')
plt.ylabel("Symbol Error Rate (SER)")
plt.title(f"SER vs Δp_tot·T (SNR=18 dB)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_formatter(NullFormatter())
plt.legend()
plt.tight_layout()
plt.show()