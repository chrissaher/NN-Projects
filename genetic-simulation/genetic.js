function compare(a, b, EPS=1e-09){
  if (a - b > EPS)
    return 1;
  if (b - a > EPS)
    return -1
  return 0
}

class Ball {
  constructor(x=0, y=0, steps = 30){
    this.x = x;
    this.y = y;
    var color = Math.round(0xffffff * Math.random());
    this.r = color >> 16;
    this.g = color >> 8 & 255;
    this.b = color & 255;

    var d = [-1, 0, 1];
    var dx = [];
    var dy = [];
    for(var i = 0; i < steps; i++) {
      var ranX = Math.floor(Math.random() * 3);
      var ranY = Math.floor(Math.random() * 3);
      dx.push(d[ranX]);
      dy.push(d[ranY]);
    }

    this.dx = dx;
    this.dy = dy;
  }

  getColor(){
    return [this.r, this.g, this.b];
  }

  getPosition(){
    return [this.x, this.y]
  }

  getVelocity(){
    return [this.dx, this.dy]
  }
}

class Genetic{
  constructor(size=5){
    this.population = this.generate(size);

  }
   generate(population_size){
     var population = [];
     for(var i = 0; i < population_size; i++) {
       population.push(new Ball());
     }
     return population;
   }

   mutate(ball, mutate_prob = 0.3) {
     var d = [-1, 0, 1];
     var nball = new Ball();
     var d = ball.getVelocity();
     var len = d[0].length;
     for(var i = 0; i < len; ++i) {
       var prob = Math.random();
       if (compare(prob, mutate_prob) == 1) {
         var npos = ((Math.random() > 0.5) + 1) % 3;
         nball.dx[i] = d[(i + npos) % 3];
         nball.dy[i] = d[(i + npos) % 3];
       }
       nball.dx[i] = ball.dx[i]
       nball.dy[i] = ball.dy[i]
     }
     return nball;
   }

   crossover(parent1, parent2){
     var vParent1 = parent1.getVelocity();
     var vParent2 = parent2.getVelocity();
     var len = vParent1[0].length
     var position = Math.random() * len;

     var dx = []
     var dy = []
     for(var i = 0; i < len; i++) {
       if (i < position) {
         dx.push(vParent1[0][i])
         dy.push(vParent1[1][i])
       } else {
         dx.push(vParent2[0][i])
         dy.push(vParent2[1][i])
       }
     }
     var child = new Ball();
     child.dx = dx;
     child.dy = dy;
     return child;
   }

   fitness(ball, timelimit=30, limitX = 27, limitY = 27, endX=27, endY=27, alpha = 0.6, beta = 0.4){
     var pos = ball.getPosition()
     var d = ball.getVelocity()
     var x = pos[0], y = pos[1];
     var dx = d[0], dy = d[1];
     var t = 0;
     for (; t < timelimit; ++t) {
       var nx = x + dx[t], ny = y + dy[t];
       if (nx < 0 || nx > limitX || ny < 0 || ny > limitY){
         break;
       }
       x = nx;
       y = ny;
       if(x == endX && y == endY){
         break;
       }
     }
     var distance_left = (endX - x) + (endY - y);
     // cost to miminize
     var fitness_cost = alpha * distance_left + beta * t;
     return -fitness_cost;
   }

   select_elite(population, fitnesses){
     var elite = 0;
     for(var i = 0; i < population.length; i++) {
       if(compare(fitnesses[i], fitnesses[elite]) == 1) {
         elite = i;
       }
     }
     return population[elite];
   }

   select(population, fitnesses){
     var len = population.length;
     var total = fitnesses.reduce(function(a, b) {
       return a + b;
     }, 0);
     var lucky = Math.random * total;
     var tempSum = 0;
     var parent1 = [];
     var fitness1 = 0;
     for(var i = 0; i < len; ++i) {
       tempSum += fitnesses[i];
       if(compare(tempSum, lucky) == 1) {
         parent1.push(population[i]);
         population.splice(i, 1);
         fitness1 = fitnesses[i];
         fitnesses.splice(i, 1);
         break;
       }
     }
     tempSum = 0;
     total = fitnesses.reduce(function(a, b) {
       return a + b;
     }, 0);
     lucky = Math.random() * total;
     for(var i = 0; i < population.length; ++i) {
       tempSum += fitnesses[i];
       if(compare(tempSum, lucky) == 1) {
         parent2 = population[i];
         population.concat(parent1);
         fitnesses.concat(fitness1);
         return [parent1, parent2];
       }
     }
   }

   createPopulation(population, fitnesses, mutate_prob=0.3){
     var len = population.length;
     var new_pop = [];
     new_pop.push(this.select_elite(population, fitnesses));
     while(new_pop.length < len) {
       var parents = this.select(population, fitnesses);
       var child = this.crossover(parents[0], parents[1]);
       new_pop.push(this.mutate(child, mutate_prob));
     }
     return new_pop;
   }

   sort(population, fitnesses){
     /*
     * Since population is short
     * a basic sort won't affect on execution time.
     */
     var len = population.length;
     for(var i = 0; i < len - 1; ++i ) {
       for(var j = i + 1; j < len; ++j) {
         if (compare(fitnesses[i], fitnesses[j]) == 1) {
           var aux = fitness[i];
           fitnesses[i] = fitnesses[j];
           fitnesses[j] = aux;

           aux = population[i];
           population[i] = population[j];
           population[j] = aux;
         }
       }
     }
   }

   simulate(max_generations = 50) {
     var generation = 0;
     var population = this.population;
     var fitnesses = [];
     for(var i = 0; i < population.length; ++i) {
       this.fitnesses.push(fitness(population[i]));
     }
     console.log("Generation %d: Fitness: %.05f", generation, this.fitness(this.select_elite));
     while(generation < max_generations) {
       generations += 1;
       population = this.createPopulation(population, fitnesses);
       fitnesses = [];
       for(var i = 0; i < population.length; ++i) {
         this.fitnesses.push(fitness(population[i]));
       }
       console.log("Generation %d: Fitness: %.05f", generation, this.fitness(this.select_elite));
     }
     return population
   }
}
