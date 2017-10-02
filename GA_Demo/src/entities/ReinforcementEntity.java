package entities;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.joml.Vector2i;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import actions.Action;
import logging.Logger;
import logging.Logger.Category;
import map.TileMap;
import map.TileType;

public class ReinforcementEntity implements Comparable<ReinforcementEntity> {

	private static final int INITIAL_LIFE = 15;
	private static final int FOOD_LIFE = 5;
	private static final int VIEW_RANGE = 5;
	private static final int BATCH_SIZE = 32;
	private static final int MAX_MEMORIES = 1000000;
	private static final int FEATURES = ((((2 * VIEW_RANGE + 1) * (2 * VIEW_RANGE + 1) - 1) * 4) + 1);
	private static final int SEED = 42;
	
	private Vector2i position;
	private int fitness = 0;
	private int life = INITIAL_LIFE;
	private boolean dead = false;
	private float alpha = 0.04f;
	private float discount = 0.97f;
	private float exploration = 1;
	private float eDecay = 0.00001f;
	private float lives = 0;
	
	private Random rand;
	private MemoryStorage memories;
	MultiLayerNetwork net;

	public ReinforcementEntity() {
		rand = new Random(SEED);
		memories = new MemoryStorage(MAX_MEMORIES, rand);
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(SEED).iterations(1)
				.weightInit(WeightInit.XAVIER).learningRate(alpha)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
				.layer(0,
						new DenseLayer.Builder().nIn((((2 * VIEW_RANGE + 1) * (2 * VIEW_RANGE + 1) - 1) * 4) + 1)
								.nOut(128).activation(Activation.RELU).build())
				.layer(1, new DenseLayer.Builder().nIn(128).nOut(64).activation(Activation.RELU).build())
				.layer(2, new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.RELU).build())
				.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
						.nIn(32).nOut(8).build())
				.backprop(true).pretrain(false).build();
		net = new MultiLayerNetwork(conf);
		net.init();
	}

	public void update(TileMap map) {

		INDArray state = generateState(map);
		int action;
		if (rand.nextFloat() > exploration) {
			action = net.predict(state)[0];
		} else {
			action = rand.nextInt(8);
		}
		exploration -= eDecay;

		Vector2i movePos = new Vector2i(position.x, position.y);
		switch (action) {
		case 0:
			movePos.add(-1, 1);
			break;
		case 1:
			movePos.add(0, 1);
			break;
		case 2:
			movePos.add(1, 1);
			break;
		case 3:
			movePos.add(-1, 0);
			break;
		case 4:
			movePos.add(1, 0);
			break;
		case 5:
			movePos.add(-1, -1);
			break;
		case 6:
			movePos.add(0, -1);
			break;
		case 7:
			movePos.add(1, -1);
			break;
		}
		TileType moveTile = map.getTile(movePos.x(), movePos.y());
		float reward = 0;
		if (moveTile == TileType.FOOD) {
			life += FOOD_LIFE;
			if (life > INITIAL_LIFE) {
				life = INITIAL_LIFE;
			}
			reward = 1;
		}
		if (moveTile == TileType.EMPTY || moveTile == TileType.FOOD) {
			map.setTile(position.x(), position.y(), TileType.EMPTY);
			position.setComponent(0, movePos.x());
			position.setComponent(1, movePos.y());
			map.setTile(position.x(), position.y(), TileType.ENTITY);
			Logger.debug(this.toString() + " position: " + position.toString(), Category.ENTITIES);
		}

		life--;
		fitness++;

		INDArray newState = null;

		if (life <= 0) {
			dead = true;
			Logger.debug(this + " Died", Category.ENTITIES);
		} else {
			newState = generateState(map);
		}
		
		memories.store(state, action, reward, newState);

	}

	private INDArray generateState(TileMap map) {
		TileType currentTile;
		List<Double> state = new ArrayList<>();

		for (int viewY = position.y() - VIEW_RANGE; viewY <= position.y() + VIEW_RANGE; viewY++) {
			for (int viewX = position.x() - VIEW_RANGE; viewX <= position.x() + VIEW_RANGE; viewX++) {
				if ((!(viewX == position.x() && viewY == position.y()))) {
					if (map.outOfRange(viewX, viewY)) {
						for (TileType type : TileType.values()) {
							state.add(0.0);
						}
					} else {
						currentTile = map.getTile(viewX, viewY);
						for (TileType type : TileType.values()) {
							if (type.equals(currentTile)) {
								state.add(1.0);
							} else {
								state.add(0.0);
							}
						}
					}
				}
			}
		}
		state.add((double) life);
		return Nd4j.create(state.stream().mapToDouble(d -> d).toArray());
	}

	public void reset() {
		life = INITIAL_LIFE;
		fitness = 0;
		dead = false;
		if(lives > 0)learn();
		lives++;
	}

	private void learn() {
		INDArray data = null;
		INDArray labels = null;
		for(Memory m : memories.batch(BATCH_SIZE)) {
			float target;
			
			INDArray state = m.getState();
			float reward = m.getReward();
			INDArray newState = m.getNewState();
			
			if(m.died()) {
				target = reward;
			}
			else {
				int newAction = net.predict(newState)[0];
				target = reward + discount * net.output(newState).getFloat(newAction);
			}
			if(data == null) {
				data = state.dup();
				labels = net.output(state).putScalar(m.getAction(), target);
			}
			else {
				data = Nd4j.vstack(data,state);
				labels = Nd4j.vstack(labels,net.output(state).putScalar(m.getAction(), target));
			}

		}
		net.fit(data,labels);
	}

	public void setPosition(Vector2i position) {
		this.position = position;
		Logger.debug(this.toString() + " position: " + position.toString(), Category.ENTITIES);
	}

	public Vector2i getPosition() {
		return position;
	}

	public int getFitness() {
		return fitness;
	}

	public boolean isDead() {
		return dead;
	}

	public void kill() {
		dead = true;
	}

	@Override
	public int compareTo(ReinforcementEntity entity) {
		if (entity.getFitness() > fitness) {
			return 1;
		} else if (entity.getFitness() == fitness) {
			return 0;
		} else {
			return -1;
		}
	}

	public void printStats() {
		Logger.info("Fitness: " + fitness);
	}

}
