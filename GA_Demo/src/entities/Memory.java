package entities;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Memory {
	private final INDArray state;
	private final int action;
	private final float reward;
	private final INDArray newState;
	
	public Memory(INDArray state, int action,float reward,INDArray newState)
	{
		this.state = state;
		this.action = action;
		this.reward = reward;
		this.newState = newState;
	}
	
	public boolean died() {
		if(state.getInt(state.length()-1) > 1) return false;
		else return !(reward==1);
	}

	public INDArray getState() {
		return state;
	}

	public int getAction() {
		return action;
	}

	public float getReward() {
		return reward;
	}

	public INDArray getNewState() {
		return newState;
	}
	
}