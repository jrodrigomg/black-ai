class Enviroment{
    constructor(){
        //Para las señales
        this.events = [];
        let self = this;
        //El websocket
        this.ws = new WebSocket("ws://localhost:9090");

        this.ws.onopen= function(){
            console.log("Conectado al ws" + " :) ")
        };


        
        this.ws.onmessage = function(evt)
        {
            console.log(evt.data)

            //Lo que venga aqui será una acción a tomar O para pedir el estado del juego.
            let action = evt.data;
            // En base a getVector() de game_manager.js...
            //     0:  STAND
            //     1:  HIT
            switch(action)
            {
                case "HIT":
                    console.log("hit")
                    $('#hit').trigger("click")
                    break;
                case "STAND":
                    console.log("stand")
                    $('#stand').trigger("click")
                    break;
                case "START":
                    self.emit("startGame")
                    break;
                case "STATE":
                    self.emit("getState");
                    break;
            }

        };
        
        
        this.ws.onclose = function()
        {
            // websocket is closed.
            console.log("connection closed");
        };
    }
}

Enviroment.prototype.sendData = function(data)
{
    this.ws.send(data);
};


//Emitir señales
Enviroment.prototype.emit = function (event, data) {
    var callbacks = this.events[event];
    if (callbacks) {
        callbacks.forEach(function (callback) {
            callback(data);
        });
    }
};

//Recibir las señales.
Enviroment.prototype.on = function (event, callback) {
    if (!this.events[event]) {
        this.events[event] = [];
    }
    this.events[event].push(callback);
};
