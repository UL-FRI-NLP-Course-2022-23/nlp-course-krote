import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = {
    'Model Name': ['Backtranslation', 'Model (128 epoch, GPT data)', 'Model (384 epoch, Backtranslation data)', 'GPT Sentences'],
    'Preserving semantic meaning score': [2.48, 1.96, 1.34, 2.4],
    'Language fluency Score': [2.62, 1.63, 1.52, 2.36],
    'Variation in word choice and grammar Score': [1.72, 1.55, 1.10, 2.52]
}
df = pd.DataFrame(data)

# Set bar width
barWidth = 0.20

# Set position of bars on X axis
r1 = np.arange(len(df['Preserving semantic meaning score']))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, df['Preserving semantic meaning score'], color='#add8e6',
        width=barWidth, edgecolor='grey', label='Preserving semantic meaning')
plt.bar(r2, df['Language fluency Score'], color='#f08080',
        width=barWidth, edgecolor='grey', label='Language fluency')
plt.bar(r3, df['Variation in word choice and grammar Score'], color='#98fb98',
        width=barWidth, edgecolor='grey', label='Variation in word choice and grammar')

# Adding xticks
plt.xlabel('Models', fontweight='bold', fontsize=15)
plt.ylabel('Score', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(df['Preserving semantic meaning score']))],
           df['Model Name'], rotation=45, ha='right')

# Create legend & Show graphic
plt.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('scores.png')
