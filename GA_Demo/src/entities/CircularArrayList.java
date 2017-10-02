package entities;

import java.util.ArrayList;

public class CircularArrayList<E> extends ArrayList<E> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private final int MAX_ENTRIES;
	private int entries = 0;
	private boolean full = false;
	
	public CircularArrayList(int capacity){
		super(capacity);
		MAX_ENTRIES = capacity;
	}

	@Override
	public boolean add(E e) {
		if(entries >= MAX_ENTRIES) {
			full = true;
		}
        return full?set(entries, e) != null:super.add(e);
    }
	
}
