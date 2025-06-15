import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.preprocessing import TransactionEncoder
from apyori import apriori

# read file and drop the first column (item counts)
df = pd.read_csv('groceries - groceries.csv')
df = df.iloc[:, 1:]

# list of lists
transactions = []
for i in range(len(df)):
    row = df.iloc[i].dropna().tolist()
    transactions.append(row)

# run apriori algo
rules = apriori(transactions, min_support=0.01, min_confidence=0.5, min_lift=1)
rules = list(rules)

# print out rules
for r in rules[:10]:  # only top 10
    print("RULE:", list(r.items))
    for stat in r.ordered_statistics:
        if stat.items_base and stat.items_add:
            print(f"  IF {list(stat.items_base)} => {list(stat.items_add)}")
            print(f"     confidence: {stat.confidence:.2f} | lift: {stat.lift:.2f}")
    print("-" * 40)

# encode
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# basic clustering without scaling
pca = PCA(n_components=2)
reduced = pca.fit_transform(df_encoded)
kmeans = KMeans(n_clusters=5, random_state=1)
labels = kmeans.fit_predict(reduced)

plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10)
plt.title('Cluster View (Raw)')
plt.grid(True)
plt.show()

# now with scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

pca2 = PCA(n_components=2)
reduced2 = pca2.fit_transform(df_scaled)
labels2 = KMeans(n_clusters=5, random_state=1).fit_predict(reduced2)

plt.scatter(reduced2[:, 0], reduced2[:, 1], c=labels2, cmap='tab10', s=10)
plt.title('Cluster View (Scaled)')
plt.grid(True)
plt.show()
