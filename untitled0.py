<!DOCTYPE html>
<html>
    <head>
        <title>Result Page </title>
        <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
    </head>
    <body >
        <body background="ground.jpg">
        <h1> <marquee style="background-color: cadetblue;"><strong>Winner Team</strong></marquee></h1>
    </body>
    
       <h1 style="color:lightseagreen;"> Winner team between- <b><span id="result"></span></b>  & <b><span id="2"></span></b> will be{{my_prediction}}</h1>
       <input type="submit" value="Result">
        <!--<script>
            document.getElementById("result").innerHTML=localStorage.getItem("ddvalue1");
            document.getElementById("2").innerHTML=localStorage.getItem("ddvalue2");
        </script>--
    
</html>
