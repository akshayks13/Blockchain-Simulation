import hashlib
import time
import json

# Deque Class
class Deque:
    '''
        The Deque (Double-Ended Queue) is a data structure that allows insertion and deletion at both ends: front and rear. 
        It maintains a circular array to efficiently manage wrap-around operations when elements are added or removed.

        Complexity Overview:
            Insertions and deletions at both ends take O(1) time due to the use of a circular array.
            Accessing front and rear elements takes O(1) as they are directly indexed.
            Space complexity is O(N) where N is the capacity of the deque.
    '''

    def __init__(self, capacity=100) -> None:
        '''
            This initializes the deque with a given capacity (default 100). The start and end pointers are initialized to -1, indicating the deque is empty. An array of size capacity is allocated to hold the deque elements.
            Time Complexity:
            Initialization Time: O(N), where N is the capacity of the deque (for array allocation).
        '''
        self.start = -1
        self.end = -1
        self.size = 0
        self.capacity = capacity
        self.deque = [0] * self.capacity

    def is_empty(self):
        '''
            Checks if the deque is empty by comparing the size attribute to zero.
            Time Complexity:
            Time: O(1), constant time check.
        '''
        return self.size == 0

    def is_full(self):
        '''
            Checks if the deque is full by comparing the size attribute with capacity.
            Time Complexity:
            Time: O(1), constant time check.
        '''
        return self.size == self.capacity

    def insert_front(self, x):
        '''
            Inserts an element x at the front of the deque. If the deque is empty, both start and end are set to 0. Otherwise, start is decremented using the circular array concept.
            Time Complexity:
            Time: O(1), direct index update and insertion at the front.
        '''
        if self.is_full():
            print("Deque is full")
            return
        
        if self.is_empty():
            self.start = 0
            self.end = 0
        else:
            self.start = (self.start - 1 + self.capacity) % self.capacity
        
        self.deque[self.start] = x
        self.size += 1

    def insert_rear(self, x):
        '''
            Inserts an element x at the rear of the deque. If the deque is empty, both start and end are set to 0. Otherwise, end is incremented circularly.
            Time Complexity:
            Time: O(1), direct index update and insertion at the rear.
        '''
        if self.is_full():
            print("Deque is full")
            return
        
        if self.is_empty():
            self.start = 0
            self.end = 0
        else:
            self.end = (self.end + 1) % self.capacity
        
        self.deque[self.end] = x
        self.size += 1

    def delete_front(self):
        '''
            Removes and returns the element at the front of the deque. If the deque becomes empty after deletion, both start and end are reset to -1.
            Time Complexity:
            Time: O(1), direct index update and deletion at the front.
        '''
        if self.is_empty():
            print("Deque is empty")
            return None
        
        deleted_element = self.deque[self.start]
        if self.start == self.end:
            self.start = -1
            self.end = -1
        else:
            self.start = (self.start + 1) % self.capacity
        
        self.size -= 1
        return deleted_element

    def delete_rear(self):
        '''
            Removes and returns the element at the rear of the deque. If the deque becomes empty after deletion, both start and end are reset to -1.
            Time Complexity:
            Time: O(1), direct index update and deletion at the rear.
        '''
        if self.is_empty():
            print("Deque is empty")
            return None
        
        deleted_element = self.deque[self.end]
        if self.start == self.end:
            self.start = -1
            self.end = -1
        else:
            self.end = (self.end - 1 + self.capacity) % self.capacity
        
        self.size -= 1
        return deleted_element

    def get_front(self):
        '''
            Returns the element at the front of the deque without removing it. If the deque is empty, it returns None.
            Time Complexity:
            Time: O(1), direct access to the front element.
        '''
        if self.is_empty():
            print("Deque is empty")
            return None
        return self.deque[self.start]

    def get_rear(self):
        '''
            Returns the element at the rear of the deque without removing it. If the deque is empty, it returns None.
            Time Complexity:
            Time: O(1), direct access to the rear element.
        '''
        if self.is_empty():
            print("Deque is empty")
            return None
        return self.deque[self.end]

    def length(self):
        '''
            Returns the number of elements currently present in the deque.
            Time Complexity:
            Time: O(1), direct access to the size attribute.
        '''
        return self.size

    def print_deque(self):
        '''
            Prints all elements in the deque from the front to the rear, iterating circularly over the array. If the deque is empty, it prints a message.
            Time Complexity:
            Time: O(N), where N is the number of elements in the deque.
        '''
        if self.is_empty():
            print("Deque is empty")
            return
        
        index = self.start
        for _ in range(self.size):
            print(self.deque[index], end=' ')
            index = (index + 1) % self.capacity

# MaxHeap Class
class MaxHeap:
    '''
        The MaxHeap class represents a priority queue where elements are stored such that the highest-priority element is always at the root of the binary heap. 
        This implementation is designed for use cases like transaction processing, where each transaction has a priority and the highest-priority transaction should be served first. 
        The heap is represented as an array, and all operations ensure that the max-heap property is maintained, i.e., 
        every parent node's priority is greater than or equal to its children's priorities.

        Complexity Overview:
            Insertions involve "heapify up" operations, which take O(log N) time.
            Deletions (extracting the maximum) involve "heapify down" operations, which also take O(log N).
            Heap sort performs O(N log N) operations.
            Space complexity is O(N) where N is the number of transactions in the heap.
    '''
    def __init__(self):
        """ Initializes the max-heap as an empty list.
            Time Complexity:
            Time: O(1), constant time initialization.
        """
        self.heap = []

    def parent(self, index):
        """
            Returns the index of the parent of a given node at index.
            Time Complexity:
            Time: O(1), direct computation.
        """
        return (index - 1) // 2

    def left_child(self, index):
        """
            Returns the index of the left child of a node at index.
            Time Complexity:
            Time: O(1), direct computation.
        """
        return 2 * index + 1

    def right_child(self, index):
        """
            Returns the index of the right child of a node at index.
            Time Complexity:
            Time: O(1), direct computation.
        """
        return 2 * index + 2

    def insert(self, transaction):
        """
             Inserts a new transaction into the heap and ensures that the max-heap property is maintained by calling heapify_up.
            Time Complexity:
            Time: O(log N), since insertion may require "bubbling up" the new element, which involves moving up the tree height (log N).
        """
        self.heap.append(transaction)
        self.heapify_up(len(self.heap) - 1)

    def heapify_up(self, index):
        """
            Ensures the max-heap property is maintained after insertion by comparing the inserted element with its parent and swapping if needed. This process continues until the property is restored.
            Time Complexity:
            Time: O(log N), the height of the heap is log N, so at most log N swaps are required.
        """
        while index > 0 and self.heap[self.parent(index)].priority < self.heap[index].priority:
            self.heap[index], self.heap[self.parent(index)] = self.heap[self.parent(index)], self.heap[index]
            index = self.parent(index)

    def extract_max(self):
        """
            Removes and returns the maximum element (root) from the heap. After removal, the last element in the heap is moved to the root, and the heapify_down process restores the max-heap property.
            Time Complexity:
            Time: O(log N), extracting the maximum element requires "heapifying down," which takes log N time.
        """
        if len(self.heap) == 0:
            return None

        root = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()

        if self.heap:
            self.heapify_down(0)

        return root

    def heapify_down(self, index):
        """
            Ensures the max-heap property is maintained after removing the root. The root is compared with its children, and swaps occur if necessary.
            This process continues until the heap property is restored.
            Time Complexity:
            Time: O(log N), at most log N comparisons and swaps are needed as the element may travel down the height of the heap.
        """
        largest = index
        left = self.left_child(index)
        right = self.right_child(index)

        if left < len(self.heap) and self.heap[left].priority > self.heap[largest].priority:
            largest = left
        if right < len(self.heap) and self.heap[right].priority > self.heap[largest].priority:
            largest = right

        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self.heapify_down(largest)

    def get_max(self):
        """
            Returns the maximum element (root) without removing it from the heap.
            Time Complexity:
            Time: O(1), direct access to the root element.
        """
        return self.heap[0] if len(self.heap) > 0 else None

    def print_heap(self):
        """
            Prints the current state of the heap by converting each transaction into a dictionary format for better visualization.
            Time Complexity:
            Time: O(N), where N is the number of elements in the heap, as it iterates over all elements.
        """
        print([tx.to_dict() for tx in self.heap])

    def heap_sort(self):
        """
            Sorts all elements in the heap in decreasing order based on priority. It repeatedly extracts the maximum element and inserts it into a sorted list.
            Time Complexity:
            Time: O(N log N), since extracting the maximum takes O(log N) and is repeated N times.
        """
        sorted_transactions = []
        while self.heap:
            sorted_transactions.insert(0, self.extract_max())
        return sorted_transactions

    def convert_to_maxheap(self, transactions):
        """
            Converts an arbitrary list of transactions into a valid max-heap. The function uses a bottom-up approach to heapify the list.
            Time Complexity:
            Time: O(N), converting a list into a heap using the bottom-up heapify approach takes linear time.
        """
        self.heap = transactions[:]
        for i in range((len(self.heap) // 2) - 1, -1, -1):
            self.heapify_down(i)

# Transaction Class 
class Transaction:
    '''
        The Transaction class represents a financial transaction between two parties, including additional metadata like priority, timestamp, and a unique identifier.
    '''
    def __init__(self, sender, receiver, amount, priority=1):
        '''
            Setting attributes like sender, receiver, amount, and priority is O(1).
            If transaction_id is not provided, generating it involves hashing, which is O(N) where N is the length of the input string.
        '''
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.priority = priority 

    def generate_transaction_id(self):
        """
            Generates a unique transaction ID based on sender, receiver, amount, and priority.
            The hash generation process has complexity O(N), where N is the combined length of the input string (sender, receiver, amount, priority, timestamp).
        """
        unique_string = f"{self.sender}{self.receiver}{self.amount}{self.priority}{self.timestamp}"
        return hashlib.sha256(unique_string.encode()).hexdigest()

    def to_dict(self):
        '''
            Converting the transaction object to a dictionary involves constant time as it is just returning predefined fields.
            Time Complexity : O(1)
        '''
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "priority": self.priority
        }

# Merkle Tree class 
class MerkleTree:
    '''
        The MerkleTree class implements a Merkle tree, a data structure commonly used in blockchain systems for efficiently and securely verifying the integrity of a collection of data (in this case, transactions).
        The tree's leaves are transaction hashes, and the non-leaf nodes are hashes of their respective children.
        This structure allows efficient verification of whether a transaction is part of the tree (through Merkle proofs), without needing to store or transmit all transactions.
    '''
    def __init__(self, transactions):
        '''
            The constructor initializes the MerkleTree with a list of transactions and calculates the Merkle root by calling calculate_merkle_root.
            Time Complexity : (O(N log N)) - This is due to the calculate_merkle_root() method
        '''
        self.transactions = transactions
        self.root = self.calculate_merkle_root()

    def calculate_merkle_root(self):
        '''
            The Merkle root is calculated by repeatedly hashing pairs of transaction hashes in a binary tree structure.
            In each level of the tree, half the number of nodes from the previous level are processed, so the total number of hash operations across all levels is O(N).
            There are O(log N) levels in a balanced tree, resulting in O(N log N) time complexity.
        '''
        if not self.transactions:
            return None

        # Start with the hashes of the transactions
        hashes = [self.hash_transaction(tx) for tx in self.transactions]

        while len(hashes) > 1:
            # If there's an odd number of hashes, duplicate the last one
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])

            # Pair and hash the nodes
            new_hashes = []
            for i in range(0, len(hashes), 2):
                new_hashes.append(self.hash_pair(hashes[i], hashes[i + 1]))
            hashes = new_hashes

        return hashes[0]

    def hash_transaction(self, transaction):
        '''
            Hashing a single transaction involves serializing the transaction object and hashing it using SHA-256.
            This operation runs in constant time, O(1), assuming the size of the transaction is small.
        '''
        return hashlib.sha256(json.dumps(transaction.to_dict(), sort_keys=True).encode()).hexdigest()

    def hash_pair(self, left, right):
        '''
            Hashing two strings (hashes of transactions or intermediate nodes) together is a constant-time operation, O(1).
        '''
        return hashlib.sha256((left + right).encode()).hexdigest()

    def verify_transaction(self, transaction, proof, root):
        """
            Verifies if a transaction is part of the Merkle Tree using the proof and the Merkle root.
            The verification process involves recomputing the root hash by following the Merkle proof.
            Since the proof contains O(log N) hash pairs (one per level of the tree), this method takes O(log N) time.   
        """
        current_hash = self.hash_transaction(transaction)
        
        for sibling_hash in proof:
            if current_hash < sibling_hash:
                current_hash = self.hash_pair(current_hash, sibling_hash)
            else:
                current_hash = self.hash_pair(sibling_hash, current_hash)

        return current_hash == root

    def get_merkle_proof(self, transaction):
        """
            Generates a Merkle proof for a given transaction.
            To generate the Merkle proof, the method traverses the tree to identify the sibling hashes at each level of the tree.
            For each level, it computes the sibling index and appends the sibling's hash to the proof. 
            Given that the tree has O(log N) levels and at each level we are dealing with O(N) transactions in the worst case, the complexity is O(N log N).
        """
        if not self.transactions or transaction not in self.transactions:
            return None
        
        hashes = [self.hash_transaction(tx) for tx in self.transactions]
        proof = []
        tx_index = self.transactions.index(transaction)

        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            
            sibling_index = tx_index ^ 1  # XOR with 1 to get sibling index
            proof.append(hashes[sibling_index])

            tx_index //= 2
            new_hashes = []
            for i in range(0, len(hashes), 2):
                new_hashes.append(self.hash_pair(hashes[i], hashes[i + 1]))
            hashes = new_hashes

        return proof

    def print_tree(self):
        """
            This method prints each level of the tree by calculating hashes at each level, similar to the root calculation.
            Therefore, the time complexity is O(N log N), as the entire tree structure is recalculated and printed.
        """
        if not self.transactions:
            print("Empty tree")
            return
        
        hashes = [self.hash_transaction(tx) for tx in self.transactions]
        level = 0
        
        while len(hashes) > 1:
            print(f"Level {level}: {hashes}")
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            new_hashes = []
            for i in range(0, len(hashes), 2):
                new_hashes.append(self.hash_pair(hashes[i], hashes[i + 1]))
            hashes = new_hashes
            level += 1
        
        print(f"Merkle Root: {hashes[0]}")

    def get_leaf_hashes(self):
        """
            This method returns the hashes of the leaf nodes (i.e., the transaction hashes).
            Since it only hashes each transaction once, the complexity is O(N).
        """
        return [self.hash_transaction(tx) for tx in self.transactions]

    def add_transaction(self, transaction):
        """
            Adding a transaction involves appending it to the list of transactions and recalculating the Merkle root.
            Rebuilding the tree (which involves rehashing all transactions) takes O(N log N).
        """
        self.transactions.append(transaction)
        self.root = self.calculate_merkle_root()

# Block class representing a single block in the blockchain
class Block:
    '''
        The Block class represents a fundamental unit in a blockchain. Each block stores a list of transactions, a timestamp, the hash of the previous block, and a nonce (used for Proof of Work).
        Additionally, it uses a Merkle Tree to ensure the integrity of the transactions and includes a hash to uniquely identify the block.
    '''
    def __init__(self, index, previous_hash, transactions, timestamp, nonce=0):
        '''
            The constructor initializes the block with its properties and immediately calculates the Merkle root and the block's hash.
            The Merkle root is computed using the MerkleTree class, which has a complexity of O(N log N), where N is the number of transactions in the block.
            Hash calculation takes O(1) time, so the overall complexity for initialization is dominated by the Merkle root calculation, resulting in O(N log N).
        '''
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp
        self.nonce = nonce
        self.merkle_root = self.calculate_merkle_root()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        '''
            The hash is calculated by serializing the block's content and hashing it with SHA-256.
            Assuming constant block size and a small number of fields in the block, this operation runs in constant time, O(1).
        '''
        block_content = json.dumps(self.to_dict(), sort_keys=True) + str(self.nonce)
        return hashlib.sha256(block_content.encode()).hexdigest()

    def to_dict(self):
        '''
            This method converts the block's attributes (including the list of transactions) into a dictionary format.
            Time complexity : O(N), assuming each transaction's to_dict method runs in constant time.
        '''
        return {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'transactions': [t.to_dict() for t in self.transactions],
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root,
        }

    def calculate_merkle_root(self):
        '''
            This method constructs a Merkle tree from the list of transactions and returns its root.
            The time complexity of the Merkle tree construction is O(N log N), where N is the number of transactions.
        '''
        merkle_tree = MerkleTree(self.transactions)
        return merkle_tree.root

    def mine_block(self, difficulty):
        '''
            Mining a block involves adjusting the nonce and recalculating the hash until the hash starts with a given number of zeros, as specified by the difficulty.
            The number of hashes attempted is proportional to 2^D, where D is the difficulty (the number of leading zeros required in the hash).
            The hash recalculation (calculate_hash) runs in O(1) time, but the loop iterates approximately 2^D times, making the mining process O(2^D).

            When the difficulty is set to 4, the mining process involves finding a hash where the first 4 characters are zeros (e.g., 0000xxxxxxxxx)

            Time Complexity : O(2^4) => O(16) or effectively O(1) , as the difficulty is set to 4 (for easy computation)
        '''
        difficulty = 4
        target = '0' * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"Block {self.index} mined with hash: {self.hash}")


# Blockchain class representing the whole chain
class Blockchain:
    '''
        The Blockchain class implements the core functionalities of a blockchain system, including managing transactions, mining blocks, and validating the chain's integrity.
        It integrates various components such as a priority queue for transactions (using a MaxHeap for prioritization), Merkle Trees for verifying transactions, and block mining with rewards based on difficulty.
        The class allows for flexible difficulty adjustment and transaction management.
        The blockchain ensures immutability through hashing and uses a decentralized reward system for miners.
    '''
    def __init__(self, difficulty=4, mining_reward=10, max_transactions=20):
        '''
            Initializes the blockchain, including the genesis block, mining difficulty, reward, and a pending transaction pool (using a MaxHeap for prioritization).
            Time Complexity:  O(1) for initializing, but creating the genesis block involves creating the first block with a transaction, which is O(1).
            Space Complexity: O(1), excluding the genesis block and pending transaction space.
        '''
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty
        self.mining_reward = mining_reward
        self.pending_transactions = MaxHeap()  # Using MaxHeap for prioritizing transactions
        self.max_transactions = max_transactions

    def create_genesis_block(self):
        '''
            Creates the initial block in the chain with a system-generated transaction.
            Time Complexity: O(1) since it creates a single block with one transaction.
            Space Complexity: O(1) for storing the block and transaction.
        '''
        return Block(0, "0", [Transaction("system", "genesis", 0)], time.time())

    def get_latest_block(self):
        '''
            Retrieves the latest block in the chain.
            Time Complexity: O(1) for retriving the last block in the chain 
            Space Complexity: O(1) 
        '''
        return self.chain[-1]
    
    def calculate_reward(self, difficulty):
        '''
            Dynamically adjusts the mining reward based on the difficulty level.
            Time Complexity: O(1) since it performs a constant calculation based on the difficulty.
            Space Complexity: O(1) 
        '''
        base_reward = 2
        return base_reward * (difficulty * 0.5)  # higher reward for higher difficulty

    def set_difficulty(self, difficulty):
        '''
            Updates the difficulty of mining and adjusts the mining reward accordingly.
            Time Complexity: O(1) for setting the new difficulty and adjusting the mining reward.
            Space Complexity: O(1) 
        '''
        self.difficulty = difficulty
        self.mining_reward = self.mining_reward + self.calculate_reward(difficulty)  # Update reward based on difficulty

    def create_transaction(self, transaction):
        '''
            Adds a new transaction to the pending transaction pool if space is available.
            Time Complexity: O(logN), where N is the number of transactions, as inserting into the MaxHeap takes ogN).
            Space Complexity: O(1) for adding a transaction to the heap.
        '''
        if len(self.pending_transactions.heap) < self.max_transactions:
            self.pending_transactions.insert(transaction)
        else:
            print("Transaction pool is full. Please mine the current transactions first.")

    def mine_pending_transactions(self, miner_address):
        '''
            Mines the pending transactions, rewards the miner, and appends a new block to the chain.
            Time Complexity: The process involves extracting K transactions (where K=max_transactions) from the MaxHeap (each extraction is O(logN)), creating a block, calculating the Merkle root (which is O(KlogK)), and mining the block, which depends on the difficulty D, making it O(2^D).
            * N represents the total number of pending transactions
            * K represents the number of transactions being mined in the current block
            (Mostly K == N)
            Total Time Complexity: O(KlogN+KlogK+2^D).
            Space Complexity: O(K) for storing the transactions to mine.

            O(2^D) = 1 [As D is set to 4]
        '''
        if len(self.pending_transactions.heap) == 0:
            print("No pending transactions to mine.")
            return

        # Limit transactions to max_transactions
        transactions_to_mine = []
        while len(transactions_to_mine) < self.max_transactions and len(self.pending_transactions.heap) > 0:
            transactions_to_mine.append(self.pending_transactions.extract_max())

        # Calculate miner's reward based on transactions
        additional_reward = len(transactions_to_mine) * 2
        total_reward = self.mining_reward + additional_reward

        # Add miner's reward transaction before mining
        reward_transaction = Transaction("system", miner_address, total_reward)
        transactions_to_mine.append(reward_transaction)

        # Create and mine a new block
        new_block = Block(len(self.chain), self.get_latest_block().hash, transactions_to_mine, time.time())
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

        print(f"Block mined successfully! Miner rewarded with {total_reward} coins.")

    def is_chain_valid(self):
        '''
            Consensus Mechanism

            Checks the validity of the blockchain by verifying block hashes and linkages.
            Time Complexity: O(B⋅T), where B is the number of blocks and T is the average number of transactions in a block.
            This is due to iterating over all blocks and validating their hashes.
            Space Complexity: O(1) for tracking validity.
        '''
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                print(f"Block {i} has been tampered with (invalid hash)!")
                return False

            if current_block.previous_hash != previous_block.hash:
                print(f"Block {i}'s previous hash does not match the previous block's hash!")
                return False

        return True

    def get_balance(self, address):
        '''
            Retrieves the balance of a specific user based on past transactions.
            Time Complexity: O(B⋅T), as it scans all transactions in all blocks.
            Space Complexity: O(1) for balance storage.
        '''
        balance = 0
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.sender == address:
                    balance -= transaction.amount
                if transaction.receiver == address:
                    balance += transaction.amount
        return balance

    def search_transaction(self, sender, receiver, amount):
        '''
            Searches for a specific transaction in the chain.
            Time Complexity: O(B⋅T), as it involves searching through all transactions in all blocks.
            Space Complexity: O(1).
        '''
        for block in self.chain:
            for transaction in block.transactions:
                if (transaction.sender == sender and
                    transaction.receiver == receiver and
                    transaction.amount == amount):
                    return transaction
        return None

    def display_all_transactions(self):
        '''
            Displays all transactions in the blockchain.
            Time Complexity: O(B⋅T), iterating through all transactions in all blocks.
            Space Complexity: O(1).
        '''
        for block in self.chain:
            print(f"Block {block.index}:")
            for transaction in block.transactions:
                print(f"  Transaction - Sender: {transaction.sender}, Receiver: {transaction.receiver}, Amount: {transaction.amount}")

    def user_transaction_history(self, address):
        '''
            Returns the transaction history of a user using a deque.
            Time Complexity: O(B⋅T), as it searches through all transactions to find the ones involving the user.
            Space Complexity: O(H), where H is the size of the transaction history deque.
        '''
        history = Deque()
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.sender == address or transaction.receiver == address:
                    history.insert_rear(transaction)
        return history
    
    def validate_transaction(self, transaction, merkle_tree, root):
        '''
            Validates a transaction against a Merkle Tree proof.
            Time Complexity: O(logT), where T is the number of transactions in the Merkle Tree, for proof verification.
            Space Complexity: O(1).
        '''
        proof = merkle_tree.get_merkle_proof(transaction)
        if proof and merkle_tree.verify_transaction(transaction, proof, root):
            print("Transaction is valid.")
        else:
            print("Transaction is invalid.")

    def generate_transaction_proof(self, transaction, merkle_tree):
        '''
            Generates a Merkle proof for a transaction.
            Time Complexity: O(logT), as generating the proof in the Merkle Tree involves traversing log layers.
            Space Complexity: O(logT) for storing the proof.

            where T is the number of transactions in the Merkle Tree
        '''
        proof = merkle_tree.get_merkle_proof(transaction)
        if proof:
            return proof
        else:
            print("Transaction not found in the Merkle Tree.")
            return None


# User class to hold wallet and perform transactions
class User:
    '''
        The User class is responsible for managing user data, specifically the user's name and wallet.
        Each user has a wallet, which can be used to interact with the blockchain, such as making transactions and retrieving transaction history.
        The transaction_history method interfaces with the blockchain to fetch the user's past transactions and displays them.
        Since the history is stored in a deque, the method handles iteration over the deque manually, accounting for its circular nature.
    '''
    def __init__(self, name):
        '''
            Initializes the User object by creating a user with a name and an associated wallet.
            Time Complexity: O(1) since creating a user object and initializing the wallet both happen in constant time.
            Space Complexity: O(1), assuming the wallet object is not large and no additional data structures are created.
        '''
        self.name = name
        self.wallet = Wallet(name)

    def get_wallet(self):
        '''
            Returns the user's wallet instance for access to the wallet's properties and functions.
            Time Complexity: O(1) as it simply returns a reference to the wallet object.
            Space Complexity: O(1).
        '''
        return self.wallet

    def transaction_history(self, blockchain):
        '''
            Retrieves and prints the transaction history of the user by interacting with the blockchain.
            It manually iterates over the user's transaction history, stored in a deque, taking into account its circular buffer structure.

            Time Complexity: O(H), where H is the size of the user's transaction history in the deque. Since the method iterates over all elements in the deque (of size H),each access is O(1), but there are H iterations
            Space Complexity: O(1) since the method prints transactions without storing any additional data. The iteration through the deque is done in place.
        '''
        history = blockchain.user_transaction_history(self.wallet.address)
        # Deque doesn't support direct iteration, so we manually iterate
        index = history.start
        for _ in range(history.size):
            tx = history.deque[index]
            print(f"Transaction - Sender: {tx.sender}, Receiver: {tx.receiver}, Amount: {tx.amount}")
            index = (index + 1) % history.capacity  # Move to the next element in circular deque


# Wallet class to hold balances and perform transactions
class Wallet:
    '''
        The Wallet class is designed to manage a user's cryptocurrency balance and facilitate transactions on a blockchain.
        Each wallet is identified by an address and maintains a balance of coins. The send_coins method allows users to send coins to other wallets, provided they have sufficient balance.
        The method also supports transaction prioritization. Additionally, the fund_initial_balances method is provided to allocate initial funds to multiple users' wallets, which can be used to bootstrap a blockchain simulation.
        This method is implemented as a static method because it doesn't depend on a particular wallet instance; instead, it operates on multiple users at once, funding their wallets based on external inputs.
    '''
    def __init__(self, address):
        '''
            Initializes the wallet with a unique address and a balance starting at 0.
            Time Complexity: O(1) since the initialization of a wallet object with an address and balance occurs in constant time.
            Space Complexity: O(1), as the object stores just the address and balance.
        '''
        self.address = address
        self.balance = 0

    def send_coins(self, blockchain, receiver, amount, priority=1):
        '''
            Allows the wallet to create a new transaction on the blockchain. It checks the wallet's balance to ensure sufficient funds are available,
            and if successful, it creates the transaction with the given priority.
            Time Complexity:The balance check blockchain.get_balance(self.address) runs in O(B), where B is the number of blocks in the chain, since it iterates over all the blocks to compute the balance.
            If the transaction can proceed, creating a transaction and adding it to the blockchain is O(logN) (as the blockchain uses a MaxHeap for storing pending transactions, which allows for efficient insertion).
            Overall, the complexity is O(B+logN), where B is the number of blocks, and N is the number of pending transactions.
            Space Complexity: O(1), since no significant additional data structures are created within the method.
        '''
        # Add priority when sending a transaction
        if blockchain.get_balance(self.address) < amount:
            print(f"Transaction failed: Insufficient balance. Current balance: {blockchain.get_balance(self.address)}")
            return
        transaction = Transaction(self.address, receiver, amount, priority)
        blockchain.create_transaction(transaction)
        print(f"Transaction of {amount} coins from {self.address} to {receiver} created with priority {priority}.")

    @staticmethod
    def fund_initial_balances(blockchain, users):
        '''
            A static method that takes a blockchain and a list of users as input and funds each user's wallet with an initial balance of 200 coins.
            It sends these funds from the system and mines the transactions to include them in the blockchain.

            Time Complexity:For each user, creating a transaction is O(logN), where N is the number of pending transactions.Mining the block takes O(2^D), where D is the difficulty of the mining process.
            If there are U users, the total time complexity is O(UlogN+2^D).
            Space Complexity: O(1) per transaction since no additional large data structures are used.
        '''
        for user in users:
            initial_fund = 200
            transaction = Transaction("system", user.wallet.address, initial_fund)
            blockchain.create_transaction(transaction)

        blockchain.mine_pending_transactions("system")
        

'''----Main----'''

def menu(my_blockchain, users, miners):
    while True:
        print("\nMenu Options:")
        print("1. Add User")
        print("2. Remove User")
        print("3. Add Miner")
        print("4. Remove Miner")
        print("5. View All Users")
        print("6. View All Miners")
        print("7. Create Transaction")
        print("8. Mine Pending Transactions")
        print("9. Show User Balance")
        print("10. Display All Transactions")
        print("11. View User Transaction History")
        print("12. Validate Blockchain")
        print("13. Show Pending Transactions")
        print("14. Adjust Mining Reward")
        print("15. Find User with Highest Balance")
        print("16. View Merkle Trees")
        print("17. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:  # Add User
            name = input("Enter new user's name: ")
            user_exists = False

            for user in users:
                if user.name == name:
                    user_exists = True
                    break

            if user_exists:
                print(f"User {name} already exists!")
            else:
                new_user = User(name)
                users.append(new_user)
                Wallet.fund_initial_balances(my_blockchain, [new_user])  # Fund the new user's balance
                print(f"User {name} added successfully with initial balance.")

        elif choice == 2:  # Remove User
            name = input("Enter user's name to remove: ")
            user_to_remove = None

            for user in users:
                if user.name == name:
                    user_to_remove = user
                    break

            if user_to_remove:
                users.remove(user_to_remove)
                print(f"User {name} removed successfully.")
            else:
                print(f"User {name} not found.")

        elif choice == 3:  # Add Miner
            name = input("Enter new miner's name: ")
            miner_exists = False

            for miner in miners:
                if miner.name == name:
                    miner_exists = True
                    break

            if miner_exists:
                print(f"Miner {name} already exists!")
            else:
                new_miner = User(name)
                miners.append(new_miner)
                print(f"Miner {name} added successfully.")

        elif choice == 4:  # Remove Miner
            name = input("Enter miner's name to remove: ")
            miner_to_remove = None

            for miner in miners:
                if miner.name == name:
                    miner_to_remove = miner
                    break

            if miner_to_remove:
                miners.remove(miner_to_remove)
                print(f"Miner {name} removed successfully.")
            else:
                print(f"Miner {name} not found.")


        elif choice == 5:  # View All Users
            if users:
                print("List of all users:")
                for user in users:
                    print(f"- {user.name}")
            else:
                print("No users found.")

        elif choice == 6:  # View All Miners
            if miners:
                print("List of all miners:")
                for miner in miners:
                    print(f"- {miner.name}")
            else:
                print("No miners found.")

        elif choice == 7:  # Create Transaction (with Priority)
            sender_name = input("Enter sender's name: ")
            receiver_name = input("Enter receiver's name: ")
            amount = float(input("Enter transaction amount: "))
            priority = int(input("Enter priority (higher number = higher priority): "))
            difficulty = int(input("Enter the new difficulty level: "))  # Ask for new difficulty
            sender = next((user for user in users if user.name == sender_name), None)
            receiver = next((user for user in users if user.name == receiver_name), None)

            if sender and receiver:
                my_blockchain.set_difficulty(difficulty)  # Update the difficulty
                sender.get_wallet().send_coins(my_blockchain, receiver.wallet.address, amount, priority)
                print(f"Transaction created with difficulty {difficulty} and updated mining reward {my_blockchain.mining_reward}")
            else:
                print("Invalid sender or receiver name.")

        elif choice == 8:  # Mine Pending Transactions
            miner_name = input("Enter miner's name: ")
            miner = next((miner for miner in miners if miner.name == miner_name), None)
            if miner:
                my_blockchain.mine_pending_transactions(miner.wallet.address)
            else:
                print("Miner not found.")

        elif choice == 9:  # Show User Balance
            name = input("Enter user's name: ")
            user = next((user for user in users if user.name == name), None)
            if user:
                print(f"{user.name}'s balance: {my_blockchain.get_balance(user.wallet.address)}")
            else:
                print(f"User {name} not found.")

        elif choice == 10:  # Display All Transactions (including hardcoded)
            my_blockchain.display_all_transactions()

        elif choice == 11:  # View User Transaction History
            name = input("Enter user's name: ")
            user = next((user for user in users if user.name == name), None)
            if user:
                user.transaction_history(my_blockchain)
            else:
                print(f"User {name} not found.")

        elif choice == 12:  # Validate Blockchain
            print("Blockchain valid?", my_blockchain.is_chain_valid())

        elif choice == 13:  # Show Pending Transactions
            if len(my_blockchain.pending_transactions.heap) == 0:
                print("No pending transactions.")
            else:
                print("Pending transactions:")
                for tx in my_blockchain.pending_transactions.heap:
                    print(f"  Sender: {tx.sender}, Receiver: {tx.receiver}, Amount: {tx.amount}, Priority: {tx.priority}")

        elif choice == 14:  # Adjust Mining Reward
            new_reward = float(input("Enter new mining reward: "))
            my_blockchain.mining_reward = new_reward
            print(f"Mining reward adjusted to {new_reward} coins.")

        elif choice == 15:  # Find User with Highest Balance
            richest_user = max(users, key=lambda user: my_blockchain.get_balance(user.wallet.address))
            print(f"The user with the highest balance is {richest_user.name} with a balance of {my_blockchain.get_balance(richest_user.wallet.address)}")

        elif choice == 16:  # View Merkle Trees
            print("\nMerkle Trees for Entire Blockchain:")
    
            if my_blockchain.chain:
                for block_index, block in enumerate(my_blockchain.chain):
                    print(f"\nMerkle Tree for Block {block_index}:")
                    block_merkle_tree = MerkleTree(block.transactions)
                    block_merkle_tree.print_tree()
                else:
                    print("The blockchain is empty.")

        elif choice == 17:  # Exit
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please try again.")


# Simulating blockchain with mining and transactions
if __name__ == "__main__":
    my_blockchain = Blockchain(difficulty=4, mining_reward=10)

    # Create users and miners
    users = [User(f"User{i}") for i in range(1, 6)]  # 5 users
    miners = [User(f"Miner{i}") for i in range(1, 4)]  # 3 miners

    # Fund initial balances
    Wallet.fund_initial_balances(my_blockchain, users)

    # Users create transactions
    users[0].get_wallet().send_coins(my_blockchain, users[1].wallet.address, 10)
    users[1].get_wallet().send_coins(my_blockchain, users[2].wallet.address, 20)
    users[2].get_wallet().send_coins(my_blockchain, users[3].wallet.address, 30)
    users[3].get_wallet().send_coins(my_blockchain, users[4].wallet.address, 40)

    # Printing the pending transactions - Maxheap
    my_blockchain.pending_transactions.print_heap()

    # Miners mine the transactions
    for miner in miners:
        print(f"\n{miner.name} is mining...")
        my_blockchain.mine_pending_transactions(miner.wallet.address)

    # Display user balances after mining
    for user in users:
        print(f"{user.name}'s balance: {my_blockchain.get_balance(user.wallet.address)}")

    # Validate blockchain and display transactions
    print("Blockchain valid?", my_blockchain.is_chain_valid())
    my_blockchain.display_all_transactions()

    # Show pending transactions if any
    if len(my_blockchain.pending_transactions.heap) == 0:
        print("\nNo pending transactions left.")
    else:
        print("\nPending transactions:")
        for tx in my_blockchain.pending_transactions.heap:
            print(f"  Sender: {tx.sender}, Receiver: {tx.receiver}, Amount: {tx.amount}, Priority: {tx.priority}")
    
    block = MerkleTree(my_blockchain.chain[1].transactions)
    block.print_tree()

    # Start the menu
    menu(my_blockchain, users, miners)
