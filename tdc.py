
from tdc.multi_pred import GDA

data = GDA(name='DisGeNET')

df = data.get_data()

print(df.head())