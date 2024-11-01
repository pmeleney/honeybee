//HoneyBeeGame

//Create GameVar Parameters
class GameState {
    gridSize = 32;
    screenSize = 1280;
    numGridX = this.screenSize/this.gridSize;
    numGridY = this.screenSize/this.gridSize;
    numFlowers = 20;
    flowerRefreshTurns = 20;
    hornetStartTurn = 0;
    hornetFrequency = 5;
    hornetRandomEle = 2;
}

// Create GameState Parameters
class GameVars {
    index = 0;
    numTurns = 0;
    beesGenerated = 0;
    foodCollected = 0;
    hornetCreated = false;
    queenAlive = true;
}

// Derive honeybee class from rectangle class
class Honeybee extends Phaser.GameObjects.Rectangle {
    constructor(scene, xCoord, yCoord, width, height, fillColor) {
        super(scene, xCoord, yCoord, width, height, fillColor);
        this.hasFood = false;
        this.score = 0;
        this.moved = false;
    }
}


//Derive hornet class from rectangle class
class Hornet extends Phaser.GameObjects.Rectangle {
    constructor(scene, xCoord, yCoord, width, height, fillColor) {
        super(scene, xCoord, yCoord, width, height, fillColor);
    }
    
}

//derive flower class from rectangle class
class Flower extends Phaser.GameObjects.Rectangle {
    constructor(scene, xCoord, yCoord, width, height, fillColor) {
        super(scene, xCoord, yCoord, width, height, fillColor);
        this.hasFood = true;
        this.turnHarvested = 0;
    }
    
}


// Nothing to preload
function preload () {}

function create() {
    
    this.gameState = new GameState
    this.gameVars = new GameVars

    this.grid = new Phaser.GameObjects.Grid(this, 0, 0, this.gameState.screenSize*2, this.gameState.screenSize*2, this.gameState.gridSize , this.gameState.gridSize, 0x666666,1,0x000000)
    this.add.existing(this.grid)

    function getFood(bee, flower) {
        if (!bee.hasFood && flower.hasFood) {
            bee.hasFood = true
            flower.hasFood = false
            flower.fillColor = 0x000000
            flower.turnHarvested = this.gameVars.numTurns
            bee.score += 100;
        }
    }
    
    function dropFood(bee) {
        if (bee.hasFood) {
            bee.hasFood = false
            this.gameVars.foodCollected += 1
            for (honeybee of this.groups.honeybeeUnits.children.entries) {
                honeybee.score += Math.floor(100/this.groups.honeybeeUnits.children.entries.length)
            }
        }
    }

    // Create a double size queen in the center of the 
    this.queen = this.add.rectangle(this.gameState.screenSize/2, this.gameState.screenSize/2, this.gameState.gridSize*2, this.gameState.gridSize*2, 0x9900aa);
    this.physics.world.enable(this.queen);

    // Initialize groups and subgroups
    this.groups = [];
    this.groups.honeybeeUnits = this.physics.add.group();
    this.groups.hornetUnits = this.physics.add.group();
    this.groups.flowerUnits = this.physics.add.group();

    // Add player next to queen
    this.player = new Honeybee(this, this.gameState.screenSize/2-16, this.gameState.screenSize/2-48, this.gameState.gridSize, this.gameState.gridSize, 0xffa500)
    this.add.existing(this.player)
    this.groups.honeybeeUnits.add(this.player)
    this.physics.world.enable(this.player);
    this.physics.add.overlap(this.player, this.queen, dropFood, null, this);

    this.player = new Honeybee(this, this.gameState.screenSize/2+16, this.gameState.screenSize/2-48, this.gameState.gridSize, this.gameState.gridSize, 0xffa500)
    this.add.existing(this.player)
    this.groups.honeybeeUnits.add(this.player)
    this.physics.world.enable(this.player);
    this.physics.add.overlap(this.player, this.queen, dropFood, null, this);

    // Add flowers as food sources, if flowers are too close to queen, choose a new location.
    while (this.groups.flowerUnits.children.entries.length < this.gameState.numFlowers) {
        this.xCoord = Phaser.Math.RND.between(4,(this.gameState.numGridX-4))*this.gameState.gridSize-16;
        this.yCoord = Phaser.Math.RND.between(4,(this.gameState.numGridY-4))*this.gameState.gridSize-16;

        while (this.xCoord < this.gameState.screenSize/2 + 128 && 
               this.xCoord > this.gameState.screenSize/2 - 128 &&
               this.yCoord < this.gameState.screenSize/2 + 128 &&
               this.yCoord > this.gameState.screenSize/2 - 128) {
            this.xCoord = Phaser.Math.RND.between(4,(this.gameState.numGridX-4))*this.gameState.gridSize-16;
            this.yCoord = Phaser.Math.RND.between(4,(this.gameState.numGridY-4))*this.gameState.gridSize-16;
        }
        this.flower = new Flower(this, this.xCoord, this.yCoord, this.gameState.gridSize, this.gameState.gridSize, 0x07da63)
        // add flowers to game and add to flower units group
        this.add.existing(this.flower)
        this.physics.world.enable(this.flower);
        this.groups.flowerUnits.add(this.flower)
    }

    this.physics.add.overlap(this.groups.honeybeeUnits, this.groups.flowerUnits, getFood, null, this);

    this.cursors = this.input.keyboard.createCursorKeys();
    this.keyPress = false;
    this.createHornetTurns = this.gameState.hornetFrequency + Phaser.Math.RND.between(0,this.gameState.hornetRandomEle);

}

function killHornet() {
    honeybee.score -= 1000
    this.newHornet.destroy()
}

function gameOver() {
    this.gameVars.queenAlive = false
    for (honeybee of this.groups.honeybeeUnits.children.entries) {
        honeybee.destroy()
    }
}

function createHoneybee(scene, gameState) {
    scene.newPlayer = new Honeybee(scene, gameState.screenSize/2-16, gameState.screenSize/2-48, gameState.gridSize, gameState.gridSize, 0xffa500)
    scene.add.existing(scene.newPlayer)
    scene.groups.honeybeeUnits.add(scene.newPlayer)
    scene.physics.world.enable(scene.newPlayer);
    scene.physics.add.overlap(scene.newPlayer, scene.queen, scene.dropFood, null, scene);
    if (scene.groups.hornetUnits.children.entries.length > 0) {
        scene.physics.add.overlap(scene.newPlayer, scene.newHornet, killHornet, null, scene);
    }
    
}

function createHornet(scene, gameState) {
    xory = Phaser.Math.RND.between(0,1)
    if (this.xory == 0) {
        xCoord = Math.floor(Math.random() * this.gameState.numGridX) * 32 + 16
        yCoord = Phaser.Math.RND.between(0,1) * this.gameState.screenSize + 16
    }
    // On Horizontal boundary
    else if (this.xory == 1) {
        yCoord = Math.floor(Math.random() * this.gameState.numGridY) * 32 +16
        xCoord = Phaser.Math.RND.between(0,1) * this.gameState.screenSize +16

    }
    scene.newHornet = new Hornet(scene, xCoord, yCoord, gameState.gridSize, gameState.gridSize, 0xff0000)
    scene.add.existing(scene.newHornet)
    scene.groups.hornetUnits.add(scene.newHornet)
    scene.physics.world.enable(scene.newHornet)
    for (honeybee of scene.groups.honeybeeUnits.children.entries) {
        scene.physics.add.collider(honeybee, scene.newHornet, killHornet, null, scene);
    }
    scene.physics.add.overlap(scene.newHornet, scene.queen, gameOver, null, scene);    
}



function update() {

    this.player = this.groups.honeybeeUnits.children.entries[this.gameVars.index]

    if (!(this.gameVars.numTurns == 0) && !this.gameVars.hornetCreated && ((this.gameVars.numTurns % this.createHornetTurns) == 0)){
        this.gameVars.hornetCreated = true
        createHornet(this, gameState)
    }

    if (!(this.gameVars.foodCollected) == 0 && ((this.gameVars.foodCollected % 2) == 0)) {
        while (this.gameVars.beesGenerated < this.gameVars.foodCollected/2) {
            this.gameVars.beesGenerated += 1
            createHoneybee(this, gameState)

        }
    }

    for (this.flower of this.groups.flowerUnits.children.entries) {
        if (this.gameVars.numTurns > (this.flower.turnHarvested + this.gameState.flowerRefreshTurns) && !this.flower.hasFood) {
            this.flower.hasFood = true;
            this.flower.fillColor = 0x07da63;
        }

    }

    if (this.groups.hornetUnits.children.entries.length > 0) {
        this.hornet = this.groups.hornetUnits.children.entries[0]
    }

    if (this.hornetTurn && (this.groups.hornetUnits.children.entries.length > 0)) {
        if ((Math.abs(this.hornet.x - this.gameState.screenSize/2)) >= (Math.abs(this.hornet.y - this.gameState.screenSize/2))) {
            if (this.hornet.x > this.gameState.screenSize/2) {
                this.hornet.x -= 32
            }
            else {
                this.hornet.x += 32
            }
        }
        else {
            if (this.hornet.y > this.gameState.screenSize/2) {
                this.hornet.y -= 32
            }
            else {
                this.hornet.y += 32
            }
        }
    }

    function endTurn(gameVars) {
        gameVars.index = 0
        gameVars.numTurns += 1
        if ((gameVars.numTurns % 1) == 0) {
            return [true,false]
        } else {
            return [false, false]
        }
    }
    this.hornetTurn=false


    if (this.groups.honeybeeUnits.children.entries.length > 0) {
        if (this.keyPress == 'up' && this.cursors.up.isUp) {
            this.keyPress = false
        }
        else if (this.keyPress == 'down' && this.cursors.down.isUp) {
            this.keyPress = false
        }
        else if (this.keyPress == 'right' && this.cursors.right.isUp) {
            this.keyPress = false
        }
        else if (this.keyPress == 'left' && this.cursors.left.isUp) {
            this.keyPress = false
        }

        if (this.cursors.up.isDown && !this.player.moved && !(this.keyPress == 'up')) {
            this.player.y -= 32;
            this.gameVars.index += 1
            this.keyPress = 'up'
            if (this.gameVars.index >=  this.groups.honeybeeUnits.children.entries.length) {
                values = endTurn(this.gameVars)
                this.hornetTurn = values[0]
                this.gameVars.hornetCreated = values[1]
            }
        }
        else if (this.cursors.down.isDown && !this.player.moved && !(this.keyPress == 'down')) {
            this.player.y += 32;
            this.gameVars.index += 1
            this.keyPress = 'down'
            if (this.gameVars.index >=  this.groups.honeybeeUnits.children.entries.length) {
                values = endTurn(this.gameVars)
                this.hornetTurn = values[0]
                this.gameVars.hornetCreated = values[1]
            }
        }
        else if (this.cursors.right.isDown && !this.player.moved && !(this.keyPress == 'right')) {
            this.player.x += 32;
            this.gameVars.index += 1
            this.keyPress = 'right'
            if (this.gameVars.index >=  this.groups.honeybeeUnits.children.entries.length) {
                values = endTurn(this.gameVars)
                this.hornetTurn = values[0]
                this.gameVars.hornetCreated = values[1]
            }
        }
        else if (this.cursors.left.isDown && !this.player.moved && !(this.keyPress == 'left')) {
            this.player.x -= 32;
            this.gameVars.index += 1
            this.keyPress = 'left'
            if (this.gameVars.index >=  this.groups.honeybeeUnits.children.entries.length) {
                values = endTurn(this.gameVars)
                this.hornetTurn = values[0]
                this.gameVars.hornetCreated = values[1]
            }
        }
}

}

gameState = new GameState

const config = {
    type: Phaser.AUTO,
    width: gameState.screenSize,
    height: gameState.screenSize,
    backgroundColor: "#333333",
    physics: {
        default: 'arcade',
        arcade: {
            gravity: {y: 0},
            enableBody: true,
            debug: true,
            }
        },
    scene: {
        preload,
        create,
        update
    }
}
const game = new Phaser.Game(config)
