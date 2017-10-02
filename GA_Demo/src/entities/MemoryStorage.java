package entities;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Random;
import java.util.Set;

public class MemoryStorage {
	private CircularArrayList<Memory> storage;
	private Random rand;
	
	public MemoryStorage(int memorySize, Random rand) {
		storage = new CircularArrayList<Memory>(memorySize);
		this.rand = rand;
	}
	
	public void store(INDArray state, int action,float reward,INDArray newState) {
		storage.add(new Memory(state,action,reward,newState));
	}
	
	public ArrayList<Memory> batch(int size) {
		
		Set<Integer> intSet = new HashSet<>();
        int storageSize = storage.size();
		size = Math.min(size,storageSize);
        while (intSet.size() < size) {
            int rd = rand.nextInt(storageSize);
            intSet.add(rd);
        }

        ArrayList<Memory> batch = new ArrayList<>(size);
        Iterator<Integer> iter = intSet.iterator();
        while (iter.hasNext()) {
            Memory mem = storage.get(iter.next());
            batch.add(mem);
        }
        

        return batch;
        
	}

}
