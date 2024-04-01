import json
import requests
from neo4j.v1 import GraphDatabase

class AdaptDatabase:
	def __init__(self,url='http://localhost',
				 port=8080):
		self.url = url
		self.port = port

	def getQuery(self,query,endpoint='json',timeout=300):
		try:
			resp = requests.post("%s:%d/query/%s"
								% (self.url,self.port,endpoint),
								data={"query": query},
								timeout=timeout)
			#print('Response: '+ str(resp.status_code))
			return (resp.json())
		except Exception as e:
			print("There was an error processing your query:")
			print(e)


class Neo4jDatabase:
	def __init__(self,url='bolt://localhost',
				 port=7687,user='neo4j',password='abc123'):
		self.url = url
		self.port = port
		self.user = user
		self.password = password
		self.driver = GraphDatabase.driver("%s:%s" % (url,port),
										   auth=(user, password))

	def getQuery(self,query,endpoint='cypher',timeout=300):
		try:
			with self.driver.session() as session:
				results = session.read_transaction(lambda tx: tx.run(query))
				return [{k:result[k] for k in results.keys()}
						for result in results]
		except Exception as e:
			print("There was an error processing your query:")
			print(e)
