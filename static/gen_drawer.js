// this program is from reference
// reference https://github.com/ginrou/handwritten_classifier/blob/master/static/drawer.js

window.addEventListener("load", function(){

    var canvas = $("#gen_canvas").get(0);
    var touchableDevice = ('ontouchstart' in window);

    if (canvas.getContext){

        var context = canvas.getContext('2d');

        var drawing = false;
        var prev = {};
        var re = null;

        canvas.width = 2 * $("#gen_canvas").width();
        canvas.height = 2 * $("#gen_canvas").height();
        context.scale(2.0, 2.0);

        context.lineJoin = "round";
        context.lineCap = "round";
        context.lineWidth = 20;
        context.strokeStyle = 'rgb(0,0,0)';

        $("#gen_canvas").bind('touchstart mousedown', function(e) {
            e.preventDefault();
            event = null;
            prev = getPointOnCanvas(this, event, e);
            drawing = true;
        });

        $("#gen_canvas").bind('touchmove mousemove', function(e) {
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

        $("#gen_canvas").bind('touchend mouseup mouseleave', function(e) {

            //console.log(drawing)
            drawing = false;
            //estimate(context);

        });

        var getPointOnCanvas = function(elem, windowEvent, touchEvent ) {
            return {
                x : (touchableDevice ? windowEvent.changedTouches[0].clientX : touchEvent.clientX ) - $(elem).offset().left,
                y : (touchableDevice ? windowEvent.changedTouches[0].clientY : touchEvent.clientY ) - $(elem).offset().top
            };
        };


        $("#rewrite_button").click(function(){
             context.clearRect(0,0,280,280);
         });


        $("#generate_button").click(function(){
            var img_buf = getImageBuffer(context, 28, 28);
            $.ajax({
                type:"post",
                url:"/generate",
                data: JSON.stringify({"input": img_buf}),
                contentType: 'application/json',
                success: function(result) {

                    re = result.vec[0].toString() + "\n"
                    input = document.getElementById('csv');
                    input.value += re
                    re = ""
                }
            });

        });


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


        document.querySelector('#download_button').onclick = function() {
            var text = document.querySelector('#csv').value;
            //console.log(text);
            this.href = 'data:text/plain;charset=utf-8,'
                + encodeURIComponent(text);
        };

    }
}, false);
