package map;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import entities.ReinforcementEntity;
import logging.Logger;
import logging.Logger.Category;

public class Population {

	private List<ReinforcementEntity> entities;
	private int size;

	public Population(int size) {
		this.size = size;
		entities = new ArrayList<>(size);
	}

	public void init() {
		for (int i = 0; i < size; i++) {
			entities.add(new ReinforcementEntity());
		}
	}

	public ReinforcementEntity getFittest() {
		ReinforcementEntity bestEntity;
		Collections.sort(entities);
		bestEntity = entities.get(0);

		return bestEntity;
	}

	public List<ReinforcementEntity> getFittestArray() {
		List<ReinforcementEntity> fittest = new ArrayList<>();
		Collections.sort(entities);
		int halfPopulationSize = entities.size() / 2;
		for (int i = 0; i < halfPopulationSize; i++) {
			fittest.add(entities.remove(0));
		}
		return fittest;
	}

	public float averageFitness() {
		float averageFitness = 0.0f;
		for (ReinforcementEntity entity : entities) {
			averageFitness += entity.getFitness();
		}
		averageFitness = averageFitness / entities.size();
		return averageFitness;
	}

	public int getSize() {
		return size;
	}

	public List<ReinforcementEntity> getEntities() {
		return entities;
	}
}
