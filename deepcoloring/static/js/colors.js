// vars
var canvas = _('#cs'),
    result = _('.result'),
    preview = _('.preview'),x = '',y = '',
    ctx = canvas.getContext('2d'),
    img = new Image(),
    graytocolor = _('#graytocolor'),
    goback = _('#goback'),
    color = document.getElementById("customcolor"),
    xy_str = '',
    file = _('#baseFile'),
    list_pos = new Array(),
    miribogi = _('#miribogi');
    
document.addEventListener('DOMContentLoaded', function(event){
  file.addEventListener('change', function(e){
    readImage( file ); 
  })
  imageRender(miribogi.src);
})

miribogi.onload = function(){
  console.log("alfvvccv")
  imageRender(miribogi.src);
}

graytocolor.addEventListener('click', function(e){
  document.body.classList.add('busy-cursor');
  
  for(var i = 0; i < list_pos.length; i++){
    htr = hexToRgb(list_pos[i][2])
    True_X = list_pos[i][0] *256 /canvas.width 
    True_Y = list_pos[i][1] *256 /canvas.height
    xy_str += htr[0] + '/' + htr[1] + '/' + htr[2] + '/' + True_X + '/' + True_Y + '/'
  }
  clicks = document.getElementById('Clicks');
  
  document.getElementById('xy_str').value = xy_str; 
  clicks.submit()
})

// click function
canvas.addEventListener('click', function(e){
  // chrome
  if(e.offsetX) {
    x = e.offsetX;
    y = e.offsetY;
  }
  // firefox
  else if(e.layerX) {
    x = e.layerX;
    y = e.layerY;
  }
  console.log("x : ", x, "Y : ", y)

  useCanvas(canvas,img, function(){
    // get image data
    var True_X = (canvas.width * x) / canvas.offsetWidth 
    var True_Y = (canvas.height * y) / canvas.offsetHeight
    
    list_pos.push([True_X, True_Y, color.value])

    for(var i = 0; i < list_pos.length; i++)
    {
      ctx.fillStyle = list_pos[i][2]
      ctx.fillRect(list_pos[i][0]-5, list_pos[i][1]-5, 10, 10)
    }

  });
},false);

goback.addEventListener('click', function(e){
  useCanvas(canvas,img, function(){
    // get image data
    list_pos.pop()
    for(var i = 0; i < list_pos.length; i++)
    {
      ctx.fillStyle = list_pos[i][2]
      ctx.fillRect(list_pos[i][0]-5, list_pos[i][1]-5, 10, 10)
    }
  });
})
// canvas function
function useCanvas(el,image,callback){
  el.width = image.width; // img width
  el.height = image.height; // img height
  // draw image in canvas tag
  el.getContext('2d')
  .drawImage(image, 0, 0, image.width, image.height);

  return callback();
}
// short querySelector
function _(el){
  return document.querySelector(el);
};

function hexToRgb( hexType ){ 

  var hex = hexType.replace( "#", "" ); 
  var value = hex.match( /[a-f\d]/gi ); 

  value = hex.match( /[a-f\d]{2}/gi ); 

  var r = parseInt( value[0], 16 ); 
  var g = parseInt( value[1], 16 ); 
  var b = parseInt( value[2], 16 ); 

  rgbType = new Array(r,g,b)
  return rgbType; 
};
function imageRender(img_file){
  img.src = img_file

  canvas.width = img.width; // img width
  canvas.height = img.height;
  
  ctx.drawImage(img, 0, 0);
}


//base64로 변환해서 반환
function readImage(input) {
  var reader = new FileReader(file);
  reader.onload = function() {
    result = reader.result;
  }
  // 실패할 경우 에러 출력하기
  reader.onerror = function (error) {
    console.log('Error');
  };
  
  if ( input.files && input.files[0] ) {
    var FR= new FileReader();
    FR.onload = function(e) {
      console.log("찍었다")
      $('#miribogi').attr( "src", e.target.result );
      $('#realdata').attr( "value" , miribogi.src);
      imageRender(miribogi.src)
    };       
    FR.readAsDataURL( input.files[0] );
  }
}

