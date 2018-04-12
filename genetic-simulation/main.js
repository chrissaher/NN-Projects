'use strict'

class Controller {

  constructor(){
    this.canvas = document.getElementById('main');
    this.context = this.canvas.getContext('2d');
    this.width = 449;
    this.height = 449;
    this.limitX = 27;
    this.limitY = 27;
    this.endX = 27;
    this.endY = 27;
    this.drawGrid()
  }
  greet(){
    var msg = 'Hello budy'
    alert(msg)
  }

  drawAtPos(x, y, r=100, g=100, b=100) {
    console.log('drawAtPos');
    var radius = 5
    var nx = 8 * (2 * x + 1);
    var ny = 8 * (2 * y + 1);
    var ctx = this.context;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(nx + radius, ny);
    ctx.arc(nx, ny, radius, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle =  'rgb(' + r + ', ' + g + ', ' + b + ')';
    ctx.fill()
  }

  clearAtPos(x, y) {
    var nx = 16 * x + 1;
    var ny = 16 * y + 1;
    var w = 14
    var h = 14
    var ctx = this.context;
    ctx.beginPath();
    ctx.rect(nx, ny, w, h);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }

  drawPath(g) {
    var t = g.dx.length, it = 0;
    var x = g.x, y = g.y;
    var dx = g.dx, dy = g.dy;
    console.log(x);
    console.log(y);
    console.log(dx);
    console.log(dy);
    this.drawAtPos(x, y, g.r, g.g, g.b);
    var id = setInterval(frame, 300);
    var controller = this;
    function frame() {
      if(it == t){
        console.log(it);
        console.log(t);
        console.log('End of simulation. No more steps.');
        clearInterval(id);
      } else {
        console.log('A')
        it++;
        var nx = x + dx[it], ny = y + dy[it];
        if (nx < 0 || nx > this.limitX || ny < 0 || ny > this.limitY){
          // console.log('End of simulation. Out of bounds.')
          // clearInterval(id);
          // break;
        } else {
          controller.clearAtPos(x, y)
          x = nx;
          y = ny;
          controller.drawAtPos(x, y, g.r, g.g, g.b);
          if(x == this.endX && y == this.endY){
            console.log('End of simulation. Reached the end.')
            // break;
            clearInterval(id);
          }
        }
      }
    }
  }

  drawGrid(){
    var ctx = this.context;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0,0,this.width, this.height);
    ctx.lineWidth = 1;
    ctx.strokeRect(0,0,this.width, this.height);
    ctx.lineWidth = 0.05;
    for(var i = 0; i < 27; ++i){
      ctx.beginPath();
      ctx.moveTo(16 * (i + 1), 0);
      ctx.lineTo(16 * (i + 1), 16 * 28);
      ctx.closePath();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, 16 * (i + 1));
      ctx.lineTo(16 * 28, 16 * (i + 1));
      ctx.closePath();
      ctx.stroke();
    }
  }
}


$(() => {
	var controller = new Controller();
  var genetic = new Genetic();
  $('#btnStart').click(function(){
    for(var it in genetic.population) {
      // console.log(genetic.population[it])
      controller.drawPath(genetic.population[it]);
    }
    // controller.drawPath(genetic.population[0]);
  });
});
