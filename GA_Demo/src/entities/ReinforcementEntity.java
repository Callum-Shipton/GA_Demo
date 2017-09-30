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
	private static final int BATCH_SIZE = 2048;
	private static final int MAX_MEMORIES = 1000000;
	private static final int FEATURES = ((((2 * VIEW_RANGE + 1) * (2 * VIEW_RANGE + 1) - 1) * 4) + 1);
	private Vector2i position;
	private int fitness = 0;
	private int life = INITIAL_LIFE;
	private boolean dead = false;
	private float alpha = 0.04f;
	private float discount = 0.9f;
	private float exploration = 1;
	private float eDecay = 0.0001f;
	private Random rand = new Random();
	private INDArray states;
	private INDArray outputs;
	private int memories = 0;
	MultiLayerNetwork net;

	public ReinforcementEntity() {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(42).iterations(1)
				.weightInit(WeightInit.XAVIER).learningRate(alpha)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
				.layer(0,
						new DenseLayer.Builder().nIn((((2 * VIEW_RANGE + 1) * (2 * VIEW_RANGE + 1) - 1) * 4) + 1)
								.nOut(512).activation(Activation.RELU).build())
				.layer(1, new DenseLayer.Builder().nIn(512).nOut(128).activation(Activation.RELU).build())
				.layer(2, new DenseLayer.Builder().nIn(128).nOut(32).activation(Activation.RELU).build())
				.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
						.nIn(32).nOut(8).build())
				.backprop(true).pretrain(false).build();
		net = new MultiLayerNetwork(conf);
		net.init();
		states = Nd4j.create(MAX_MEMORIES, FEATURES);
		outputs = Nd4j.create(MAX_MEMORIES, 8);
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

		float target;

		if (life <= 0) {
			dead = true;
			Logger.debug(this + " Died", Category.ENTITIES);
			target = reward;
		} else {
			INDArray newState = generateState(map);
			int newAction = net.predict(newState)[0];
			target = reward + discount * net.output(newState).getFloat(newAction);

		}
		INDArray targetArr = net.output(state).putScalar(action, target);
		storeMemory(state, targetArr);

	}

	private void storeMemory(INDArray state, INDArray targetArr) {
		if (memories < MAX_MEMORIES) {
			states.putRow(memories, state);
			outputs.putRow(memories, targetArr);
			memories++;
		}
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
		if (memories > BATCH_SIZE) {
			DataSet memory = new DataSet(states.get(NDArrayIndex.interval(0,memories), NDArrayIndex.all()), outputs.get(NDArrayIndex.interval(0,memories), NDArrayIndex.all()));
			DataSet m = memory.sample(BATCH_SIZE,false);
				net.fit(m.getFeatures(), m.getLabels());
		}
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
