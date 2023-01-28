'''main code'''



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

n =1000
x = np.random.randn(n)
y= np.random.randn(n)**2

plt.hist2d(x,y,30,vmax=10)
plt.show()

sns.jointplot(x,y,kind='scatter',color=[.8,.8,.3]).plot_joint(sns.kdeplot)


//////////////////////////////////////////////////////////////////////////////////////////////////////////













import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
data=pd.read_csv('D://Study Materials//AI//blackbook//prac4//data.csv',header=None,names=['x','y'])
print(data)
sns.jointplot(data['x'], data['y'], kind ='reg').plot_joint(sns.scatterplot)
plt.show()
