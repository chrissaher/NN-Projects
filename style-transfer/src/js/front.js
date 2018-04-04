class NeuralStyleTransfer {

  content_cost(content, generated){
    
  }
}


$(() => {


  $('#clickme').click(function() {
    style_canvas = $('#style')[0];
    style_ctx = style_canvas.getContext('2d');
    style_src = document.getElementById('imgstyle');
    style_ctx.drawImage(style_src, x=10, y=10, width=240, height=240);

    content_canvas = $('#content')[0];
    content_ctx = content_canvas.getContext('2d');
    content_src = document.getElementById('imgcontent');
    content_ctx.drawImage(content_src, x=10, y=10, width=240, height=240);

  });
});
