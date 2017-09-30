
package map;

import display.Window;
import entities.ReinforcementEntity;
import logging.Logger;
import logging.Logger.Category;

public class Village {

	private int maxFood;

	private Population population;
	private ReinforcementEntity fittestEntity;
	private float fittestGeneration = 0;

	private int generation = 1;
	private int time = 0;

	private TileMap map;

	private static final int FOOD_DELAY = 5;
	private int foodCounter = FOOD_DELAY;

	private int moveCounter = 0;
	private static final int MOVE_DELAY = 1;

	public Village(int populationSize, int mapSize, int maxFood) {
		this.maxFood = maxFood;
		map = new TileMap(mapSize);
		population = new Population(populationSize);
	}

	public void setUp() {
		map.reset();

		population.init();
		map.spawnPopulation(population);

		createInitialFood();
	}

	public void createInitialFood() {
		for (int i = 0; i < maxFood; i++) {
			map.setEmptyTile(TileType.FOOD);
		}
	}

	public void render(Window w) {
		map.render(w);
	}

	public void update() {
		if (!map.getEntities().isEmpty()) {
			if (moveCounter <= 0) {
				time++;
				Logger.debug("Time: " + time, Category.SYSTEM);
				map.update();

				foodCounter--;
				if (foodCounter <= 0) {
					map.setEmptyTile(TileType.FOOD);
					foodCounter = FOOD_DELAY;
				}

				moveCounter = MOVE_DELAY;
			}

			moveCounter--;

		} else {
			createNextGame();
		}
	}

	public void createNextGame() {
		ReinforcementEntity fittestGenEntity = population.getFittest();
		if (fittestEntity == null || (fittestGenEntity.getFitness() > fittestEntity.getFitness())) {
			fittestEntity = fittestGenEntity;
		}
		float generationFitness = population.averageFitness();
		if (generationFitness > fittestGeneration) {
			fittestGeneration = generationFitness;
		}
		Logger.info("Generation: " + generation);
		Logger.info("Fittest Entity of Generation: ");
		fittestGenEntity.printStats();
		Logger.info("Average Fitness of Generation: " + generationFitness);
		Logger.info("Fittest Entity of all time:");
		fittestEntity.printStats();
		Logger.info("Fittest Generation of all time: " + fittestGeneration);
		time = 0;
		generation++;
		map.reset();
		createInitialFood();
		map.spawnPopulation(population);
	}
}
