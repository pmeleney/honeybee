class honeyBeeGame extends Phaser.Scene
{
    gridSize = 32
    screenSize = 1280/gridSize

    gameState = {
        score: 0,
    }

    preload () {
        this.load.image('honeybee', 'https://s3.amazonaws.com/codecademy-content/courses/learn-phaser/physics/bug_1.png');
        this.load.image('food', 'https://s3.amazonaws.com/codecademy-content/courses/learn-phaser/physics/bug_2.png');
        this.load.image('hornet', 'https://s3.amazonaws.com/codecademy-content/courses/learn-phaser/physics/bug_3.png');
        this.load.image('platform', 'https://s3.amazonaws.com/codecademy-content/courses/learn-phaser/physics/platform.png');
    }

    create () {
        this.add.grid(0, 0, gridWidth * gridSize, gridHeight * gridSize, gridSize, gridSize, 0x999999, 1, 0x666666).setOrigin(0);

        gameState.scoreText = this.add.text(gameState.screenSize-200, 100, 'Score: 0', { fontSize: '15px', fill: '#fff' })

        this.queen = this.physics.add.sprite(gameState.screenSize/2, gameState.screenSize/2, 'honeybee').setScale(1.5);

        this.player = this.physics.add.sprite(gameState.screenSize/2, gameState.screenSize/2-50, 'honeybee').setScale(.5);
        
        this.player.setCollideWorldBounds(true);

        this.physics.world.enable(this.player)
        this.physics.world.enable(this.queen)

        const hornets = this.physics.add.group();
        const foods = this.physics.add.group();

        const hornetGen = () => {
            const xory = Math.floor(Math.random() * 2)
            // On Vertical Axis
            if (xory == 0) {
                xCoord = Math.floor(Math.random() * 2) * gameState.screenSize
                yCoord = Math.random() * gameState.screenSize
            }
            else if (xory == 1) {
                yCoord = Math.floor(Math.random() * 2) * gameState.screenSize
                xCoord = Math.random() * gameState.screenSize

            }
            else {
                console.log('error')
            }
            hornets.create(x=xCoord, y=yCoord, frame='hornet')
            hornets.setVelocityX((xCoord - gameState.screenSize/2)/-20)
            hornets.setVelocityY((yCoord - gameState.screenSize/2)/-20)
        }

        const foodGen = () => {
            xCoord = Math.random() * gameState.screenSize
            yCoord = Math.random() * gameState.screenSize
            foods.create(x=xCoord, y=yCoord, frame='food')
        }

        const hornetGenLoop = this.time.addEvent({
            delay: 20000,
            callback: hornetGen,
            loop: true,
        });

        const foodGenLoop = this.time.addEvent({
            delay: 5000,
            callback: foodGen,
            loop: true,
        });

        this.physics.add.overlap(foods, this.player, eatFood, null, this);
        this.physics.add.collider(hornets, this.player, killHornet, null, this);
        this.physics.add.collider(hornets, this.queen, endGame, null, this);
        
        function eatFood(player, food) {
            food.destroy()
            gameState.score += 10;
            gameState.scoreText.setText(`Score: ${gameState.score}`)		
        }

        function killHornet(player, hornet) {
            hornet.destroy()
            gameState.score += 100;
            gameState.scoreText.setText(`Score: ${gameState.score}`)		
        }

        function endGame() {
            hornetGenLoop.destroy();
            foodGenLoop.destroy();
            this.physics.pause();

            this.add.text(280, 150, 'Game Over \n Click to Restart', { fontSize: '15px', fill: '#FFF' })
            gameState.score = 0

            this.input.on('pointerdown', () => {
                this.scene.restart();
            })
        }
        
    }

    update () {
        const cursors = this.input.keyboard.createCursorKeys();

        if(cursors.left.isDown){
            this.player.setVelocityX(-200)
            this.player.setVelocityY(0)
        } else if (cursors.right.isDown) {
            this.player.setVelocityX(200)
            this.player.setVelocityY(0)
        } else if (cursors.up.isDown) {
            this.player.setVelocityY(-200);
            this.player.setVelocityX(0)
        } else if (cursors.down.isDown) {
            this.player.setVelocityY(200);
            this.player.setVelocityX(0)
        }
        else {
            this.player.setVelocityX(0);
            this.player.setVelocityY(0);
        }
    }



    config = {
    type: Phaser.AUTO,
    width: honeyBeeGame.screenSize,
        height: honeyBeeGame.screenSize,
        backgroundColor: "333333",
        physics: {
            default: 'arcade',
            arcade: {
                gravity: {y: 0},
                enableBody: true,
                debug: false,
            }
        },
    scene: {
        honeyBeeGame
        }
    }
}

const game = new Phaser.Game(honeyBeeGame.config)
