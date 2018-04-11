

class Controller{

  receive_message(){
    $.ajax({
            url:'/api/nst',
            method:'POST',
            contentType: 'application/json',
            data: {
              name :3,
              lastname :4
            },
            success:(data) =>{
                alert(data.msg); // How to get the data
                this.data = data;
              }
            });
  }
}

$(() => {
  con = new Controller();
});
