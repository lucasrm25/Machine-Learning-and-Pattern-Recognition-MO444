import cluster

c = cluster.Cluster ()
c.readCsv()
c.preprocess ()
c.computeFeatures()
