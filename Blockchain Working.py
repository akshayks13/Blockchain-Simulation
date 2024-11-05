import time
import hashlib

"""
Key components of the implementation include:
-Block: Represents a single block in the blockchain, containing transactions, a hash of the previous block, and a method for mining the block through proof-of-work.
-Transaction: Represents a financial transaction between two parties, encapsulating details such as sender, recipient, and amount, along with a unique transaction ID.
-MerkleNode and Merkle Tree: Implemented to facilitate efficient verification of transactions within a block, enhancing security and integrity by allowing a compact representation of all transactions.
-Blockchain: Represents the entire chain of blocks, managing block creation, transaction handling, and chain validation to ensure the integrity of the blockchain.
-Node and Network: Simulate the behavior of decentralized network nodes, capable of receiving, validating, and propagating blocks within the network.

Functionalities covered:
1)Creating transactions 
2)Mining blocks
3)Verifying the integrity of the blockchain
4)Simulating network interactions between nodes. 
5)Incorporates a basic consensus mechanism, employing the longest chain rule to resolve conflicts among competing blockchains.
"""

# Block class
class Block: #Initializes a new block in the blockchain.
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index #position of block in the chain
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time() #optionally accepts a timestamp else default = current time since epoch
        self.nonce = 0 #number used in mining to find valid block hash , number used once
        self.merkle_root = self.calculate_merkle_root()
        self.hash = self.calculate_hash() #hash of block's content

    def calculate_merkle_root(self):# Time Complexity: O(n*log(n)) where n is the number of transactions

        """
        Calculates the Merkle root for the block's transactions.
        Returns:
        - The Merkle root of the transactions in the block.
        """
        return create_merkle_root(self.transactions).data

    def calculate_hash(self): #O(1)
        #concatenate contents of block and hash em via SHA-256
        block_content = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}{self.nonce}{self.merkle_root}"
        return hashlib.sha256(block_content.encode()).hexdigest() #data to bytes , bytes via SHA256 , then to Hex

    def mine_block(self,difficulty):# Time Complexity: O(2^difficulty), as it depends on difficulty level
        """
        Mines the block by finding a valid hash with a given difficulty.

        Parameters:
        - difficulty: The number of leading zeros required in the hash.
        """
        while self.hash[:difficulty]!='0'*difficulty:
            self.nonce += 1 
            self.hash = self.calculate_hash()
        print(f"Block mined: {self.hash}")

class Transaction:
    def __init__(self, sender, recipient, amount):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.transaction_id = hash(f"{self.sender}{self.recipient}{self.amount}")
    
    def calculate_transaction_id(self):
        # Create a string from transaction details and hash it using SHA-256
        transaction_string = f"{self.sender}{self.recipient}{self.amount}"
        return hashlib.sha256(transaction_string.encode()).hexdigest()

class MerkleNode: #Represents a node in the Merkle tree for transaction verification.
    #Data Verification efficient , O(log n)
    def __init__(self,left=None,right=None,data=None):
        self.left = left
        self.right = right
        self.data = data

def create_merkle_root(transactions):# Time Complexity: O(n*log(n)), where n is the number of transactions
 #Creates a Merkle root from a list of transactions.
    if not transactions:
        return MerkleNode(data=hash("0"))
    nodes = []
    for transaction in transactions:
        nodes.append(MerkleNode(data=hash(transaction.transaction_id)))
    while len(nodes)>1:
        if len(nodes)%2!=0:
            nodes.append(nodes[-1])
        temp_nodes=[] #stores new nodes created by combining pairs of existing nodes
        for i in range(0,len(nodes),2):
            left = nodes[i]
            right = nodes[i+1]
            temp_nodes.append(MerkleNode(left=left, right=right, data=hash(left.data+right.data)))
        nodes = temp_nodes
    return nodes[0]

class Blockchain: #Initializes a new blockchain with the genesis block and default settings.
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 4
        self.pending_transactions = []
        self.reward_for_mining = 50

    def create_genesis_block(self):
        return Block(0,"0",[],time.time()) #index,previous_hash,transactions,timestamp

    def get_last_block(self):
        return self.chain[-1]

    def mine_pending_transactions(self,miner_address):
        """
        Mines all pending transactions and creates a new block.

        Parameters:
        - miner_address: The address of the miner receiving the reward.
        """
        new_block = Block(len(self.chain),self.get_last_block().hash,self.pending_transactions)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.pending_transactions = [Transaction(None, miner_address, self.reward_for_mining)] #reward , so sender is None

    def add_transaction(self, transaction):
        """
        Adds a new transaction to the list of pending transactions.

        Parameters:
        - transaction: The transaction to be added.
        """
        self.pending_transactions.append(transaction)

    def is_chain_valid(self):
        """
        Validates the entire blockchain by checking hashes and Merkle roots.

        Returns:
        - True if the chain is valid, False otherwise.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            if current_block.hash != current_block.calculate_hash() or current_block.previous_hash != previous_block.hash: 
#check ensures that the block's content has not been altered. If the current block's hash!=hash that is recalculated => block has been tampered with
                return False
            if current_block.merkle_root != current_block.calculate_merkle_root():
                return False
        return True

class Network:
    def __init__(self):
        self.nodes = []

    def add_node(self,node):
        self.nodes.append(node)

    def traverse_blockchain(self,block):
        for node in self.nodes:
            node.receive_and_check_block(block)

class Node:
    def __init__(self,name,blockchain):
        self.name = name
        self.blockchain = blockchain

    def receive_and_check_block(self,block):
        latest_block = self.blockchain.get_last_block()
        if latest_block.hash==block.previous_hash and block.hash==block.calculate_hash():
            self.blockchain.chain.append(block)
            print(f"{self.name} received and accepted new block: {block.hash}")
        else:
            print(f"{self.name} rejected the block: {block.hash} (Invalid)")
        if not self.blockchain.is_chain_valid():#checks validity and maintains integrity
            print(f"{self.name}'s blockchain is invalid after receiving the block")
        
    def traverse_blockchain(self, block):
        for node in self.network_nodes:
            if node.receive_and_check_block(block):  # This should return True if valid
                print(f"Block successfully propagated to {node.name}.")
            else:
                print(f"{node.name} rejected the block: {block.hash} (Invalid)")


#Consensus mechanism to resolve chain conflicts
#The longest chain represents the most work done (computationally expensive), so it's more likely to be valid and honest.
#Other shorter chains are discarded.
def resolve_conflicts(blockchains):
    longest_chain = blockchains[0]
    for chain in blockchains:
        if len(chain)>len(longest_chain):
            longest_chain = chain
    return longest_chain

# Step 1: Initialize the blockchain and create the genesis block
blockchain = Blockchain()
print("Step 1: Initialized Blockchain with Genesis Block")
print(f"Genesis Block: {blockchain.chain[0].hash}")

# Step 2: Create a valid transaction
print("\nObjective 2: Creating Valid Transactions")
transactions = [
    Transaction(sender="Tarun", recipient="Aswin", amount=10),
    Transaction(sender="Aswin", recipient="Akshay", amount=6),
    Transaction(sender="Akshay", recipient="Sarvesha", amount=13),
    Transaction(sender="Sarvesha", recipient="Ram", amount=15),
]

for trans in transactions:
    print(f"Transaction Created: {trans.calculate_transaction_id()}")
    blockchain.add_transaction(trans)

print("All transactions added to pending transactions.")

# Step 3: Mine a block to add the transaction to the blockchain
print("\nObjective 3: Mining a Block with Pending Transactions")
blockchain.mine_pending_transactions(miner_address="Miner_1")
print(f"Block Mined: {blockchain.chain[-1].hash}")
if len(blockchain.chain)==2:
    print("Block mining succeeded!")
else:
    print("Block mining failed!")

# Step 4: Verify blockchain validity after adding the new block
print("\nObjective 4: Verifying Blockchain Validity")
if blockchain.is_chain_valid():
    print("Blockchain is valid.")
else:
    print("Blockchain is invalid!")

# Step 5: Calculate Merkle root for the new block
print("\nObjective 5: Verifying Merkle Root Calculation")
block = blockchain.chain[-1]
merkle_root = create_merkle_root(block.transactions)
if block.merkle_root == merkle_root.data:
    print("Merkle Root calculation successful & Verified!!")
    print(f"Merkle Root: {block.merkle_root}")
else:
    print("Merkle root calculation failed!")

# Step 6: Simulate network block propagation between two nodes
print("\nObjective 6: Simulating Network Propagation of Mined Block")

network = Network()
nodeA = Node(name="Node1",blockchain=Blockchain())
nodeB = Node(name="Node2",blockchain=Blockchain())
nodeC = Node(name="Node3",blockchain=Blockchain())

# Adding nodes to the network
network.add_node(nodeA)
network.add_node(nodeB)
network.add_node(nodeC)

nodeA.blockchain.mine_pending_transactions(miner_address="Miner_A")
nodeB.blockchain.mine_pending_transactions(miner_address="Miner_B")
nodeC.blockchain.mine_pending_transactions(miner_address="Miner_C")

# Propagating the block from node_A to node_B
network.traverse_blockchain(nodeA.blockchain.get_last_block())

# Checking if both nodes have the same chain
if len(nodeA.blockchain.chain)==len(nodeB.blockchain.chain):
    print("Block successfully propagated to Node_B.")
else:
    print("Network propagation failed!")

# Consensus mechanism to resolve chain conflicts and return the resolved chain
print("\nObjective 7: Testing Consensus Mechanism (Longest Chain Rule)")
blockchain1 = Blockchain()
blockchain1.mine_pending_transactions(miner_address="Miner_1")
print("\n\n")

blockchain2 = Blockchain()
blockchain2.mine_pending_transactions(miner_address="Miner_2")
blockchain2.mine_pending_transactions(miner_address="Miner_2")
blockchain2.mine_pending_transactions(miner_address="Miner_2")
blockchain2.mine_pending_transactions(miner_address="Miner_2")
blockchain2.mine_pending_transactions(miner_address="Miner_2")
print("\n\n")

blockchain3 = Blockchain()
blockchain3.mine_pending_transactions(miner_address="Miner_3")

resolved_chain = resolve_conflicts([blockchain1.chain,blockchain2.chain,blockchain3.chain])
print(f"The Length of the Resolved Chain is : {len(resolved_chain)}")