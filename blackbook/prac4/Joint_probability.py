import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
data=pd.read_csv('D://Study Materials//AI//blackbook//prac4//data.csv',header=None,names=['x','y'])
print(data)
sns.jointplot(data['x'], data['y'], kind ='reg').plot_joint(sns.scatterplot)
plt.show()