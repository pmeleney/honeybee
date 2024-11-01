class honeyBeeGame extends Phaser.Scene
{
    gridSize = 32;
    screenSize = 1280;
    numGridX = this.screenSize/this.gridSize;
    numGridY = this.screenSize/this.gridSize;
    playerIndex = -1;

    gameState = {
        numMoves: 0,
        score: 0,
        numBees: 1,
        foodCollected: 0,
    }

    preload () {

    }

    create () {

        this.gameState.scoreText = this.add.text(this.screenSize-200, 100, 'Score: 0', { fontSize: '15px', fill: '#FFFFFF' })
        

        this.queen = this.add.rectangle(this.screenSize/2, this.screenSize/2, 32*2, 32*2, 0xffa500)
        this.player = this.add.rectangle(this.screenSize/2-16, this.screenSize/2-48, 32,32,0xffa500);
        this.physics.world.enable(this.player)
        this.physics.world.enable(this.queen)
        this.player.body.setCollideWorldBounds(true);

        this.foods = this.physics.add.group();
        this.hornets = this.physics.add.group();

        const hornetGen = () => {
            this.xory = Math.floor(Math.random() * 2)
            // On Vertical Boundary
            if (this.xory == 0) {
                this.xCoord = Math.floor(Math.random() * 2) * this.screenSize
                this.yCoord = Math.random() * this.screenSize
            }
            // On Horizontal boundary
            else if (this.xory == 1) {
                this.yCoord = Math.floor(Math.random() * 2) * this.screenSize
                this.xCoord = Math.random() * this.screenSize

            }
            else {
                console.log('error')
            }
            this.hornet = this.add.rectangle(this.xCoord, this.yCoord, this.gridSize, this.gridSize, 0xFF0000)
            this.physics.world.enable(this.hornet)
            this.physics.add.overlap(this.hornet, this.player, killHornet, null, this);
            this.hornets.add(this.hornet)
            this.hornets.setVelocityX((this.xCoord - this.screenSize/2)/-10)
            this.hornets.setVelocityY((this.yCoord - this.screenSize/2)/-10)
        }

        const foodGen = () => {
            this.xCoord = Phaser.Math.RND.between(2,(this.numGridX-2))  * this.gridSize
            this.yCoord = Phaser.Math.RND.between(2,(this.numGridY-2))  * this.gridSize
            
            this.food = this.add.rectangle(this.xCoord, this.yCoord, this.gridSize, this.gridSize, 0x07da63)
            this.physics.world.enable(this.food)
            this.physics.add.overlap(this.food, this.player, eatFood, null, this);
            this.foods.add(this.food)
        }

        const hornetGenLoop = this.time.addEvent({
            delay: 10000,
            callback: hornetGen,
            loop: true,
        });

        const foodGenLoop = this.time.addEvent({
            delay: 1000,
            callback: foodGen,
            loop: true,
        });

        this.physics.add.overlap(this.hornets, this.player, killHornet, null, this);
        this.physics.add.overlap(this.hornets, this.queen, endGame, null, this);
        
        function eatFood(food) {
            food.destroy()
            this.gameState.score += 10;
            this.gameState.scoreText.setText(`Score: ${this.gameState.score}`)
        }

        function killHornet(player, hornets) {
            hornets.destroy()
        }

        function endGame() {
            hornetGenLoop.destroy();
            foodGenLoop.destroy();
            this.physics.pause();

            this.add.text(280, 150, 'Game Over \n Click to Restart', { fontSize: '15px', fill: '#FFF' })
            this.gameState.score = 0

            this.input.on('pointerdown', () => {
                this.scene.restart();
            })
        }
        
    }

    update () {
        const cursors = this.input.keyboard.createCursorKeys();

        function nextTurn() {
            this.playerIndex++
            // Loop over characters
            if (this.playerIndex >= this.units.length) {
                this.playerIndex = 0
            }
            // If player character (honeybee) choose how to move
            if (this.units[this.playerIndex] instanceof PlayerCharacter) {
                this.events.emit('PlayerMove', this.playerIndex)
            }
            // Is Hornet Character
            else {
                this.units[this.playerIndex].moveToQueen()

            }
        }

        if(cursors.left.isDown){
            this.player.x = this.player.x -= 3
        } else if (cursors.right.isDown) {
            this.player.x = this.player.x += 3
        } else if (cursors.up.isDown) {
            this.player.y = this.player.y -= 3
        } else if (cursors.down.isDown) {
            this.player.y = this.player.y += 3
        }
        else {
            this.player.y = this.player.y += 0
            this.player.x = this.player.x += 0
        }
    }
}

const config = {
    type: Phaser.AUTO,
    width: 1280,
    height: 1280,
    backgroundColor: "#333333",
    physics: {
        default: 'arcade',
        arcade: {
            gravity: {y: 0},
            enableBody: true,
            debug: true,
            }
        },
    scene: honeyBeeGame
}

const game = new Phaser.Game(config)
