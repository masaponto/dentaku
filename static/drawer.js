// this program is from reference
// reference https://github.com/ginrou/handwritten_classifier/blob/master/static/drawer.js

window.addEventListener("load", function(){

    var canvas = $("#canvas").get(0);
    var touchableDevice = ('ontouchstart' in window);

    if (canvas.getContext){

        var context = canvas.getContext('2d');

        var drawing = false;
        var prev = {};
        var re = null;

        canvas.width = 2 * $("#canvas").width();
        canvas.height = 2 * $("#canvas").height();
        context.scale(2.0, 2.0);

        context.lineJoin = "round";
        context.lineCap = "round";
        context.lineWidth = 20;
        context.strokeStyle = 'rgb(0,0,0)';

        $("#canvas").bind('touchstart mousedown', function(e) {
            e.preventDefault();
            event = null;
            prev = getPointOnCanvas(this, event, e);
            drawing = true;
        });

        $("#canvas").bind('touchmove mousemove', function(e) {
            if(drawing == false) return;

            e.preventDefault();
            curr = getPointOnCanvas(this, event, e);

            // draw
            context.beginPath();
            context.moveTo(prev.x, prev.y);
            context.lineTo(curr.x, curr.y);
            context.stroke();

            // update
            prev = curr;
        });

        $("#canvas").bind('touchend mouseup mouseleave', function(e) {

            // console.log(drawing)

            drawing = false;
            estimate(context);
        });

        var getPointOnCanvas = function(elem, windowEvent, touchEvent ) {
            return {
                x : (touchableDevice ? windowEvent.changedTouches[0].clientX : touchEvent.clientX ) - $(elem).offset().left,
                y : (touchableDevice ? windowEvent.changedTouches[0].clientY : touchEvent.clientY ) - $(elem).offset().top
            };
        };

        $("#delete_button").click(function(){
            context.clearRect(0,0,280,280);
            re = '';
        });

        $("#accept_button").click(function(){
            input = document.getElementById('inputText');
            input.value += re;
            re = ''
            input.focus();
            input.onchange;
            $('inputText').trigger('change');
            //document.getElementById('inputText').onchange;
            //document.getElementById('inputText').fireEvent('onchange');
            //document.getElementById('inputText').keyup();
            //input.focus();


            // if (input.createTextRange) {
            //     var range = input.createTextRange();
            //     range.move('character', input.value.length);
            //     range.select();
            // } else if (input.setSelectionRange) {
            //     input.setSelectionRange(input.value.length, input.value.length);
            // }

            //document.getElementById('inputText').focus();
            //document.getElementById('inputText').value += re;

            // var input = document.getElementById('inputText');
            // var text = input.value;
            // input.value = '';
            // input.focus();
            // text.value = text + re

            context.clearRect(0,0,280,280);

            re = '';
        });

        var estimate = function(context) {
            var img_buf = getImageBuffer(context, 28, 28);
            $.ajax({
                type:"post",
                url:"/estimate",
                data: JSON.stringify({"input": img_buf}),
                contentType: 'application/json',
                success: function(result) {
                    $("#estimated").text("This Number is " + result.estimated + " ?");
                    re = result.estimated
                }
            });
        };

        var getImageBuffer = function(context, width, height) {
            var tmpCanvas = $('<canvas>').get(0);
            tmpCanvas.width = width;
            tmpCanvas.height = height;
            var tmpContext = tmpCanvas.getContext('2d');
            tmpContext.drawImage(context.canvas, 0, 0, width, height);
            var image = tmpContext.getImageData(0,0,width,height);

            var buffer = []
            for( var i = 0; i < image.data.length; i += 4 ) {
                var sum = image.data[i+0] + image.data[i+1] + image.data[i+2] + image.data[i+3];
                buffer.push(Math.min(sum,255));
            }
            return buffer;
        };

    }
}, false);
