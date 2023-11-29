def main():
    global game_speed, x_pos_bg, y_pos_bg, points,death_count, obstacles, obstacle_distance, obstacle_width, obstacle_height

    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 14
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    obstacles = []
    
    
    #inputs
    obstacle_distance = 0
    obstacle_width = 0
    obstacle_height = 0
    normalized_inputs = np.array([0,0,0,0])
    target_options = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1],[1,0,0]])
    action = 0
    learning_rate = 0.01
    reward = 0
    ####

    model = Model(4,4,3)
    epoka = 0

    for i in range(len(normalized_inputs)):
            inputs = normalized_inputs[i]
            target = target_options[i]

            # Propagacja wprzód
            model.forward(inputs)
            
            # Oblicz funkcjê straty
            loss = model.calculate_loss(model.output, target) - reward

            

            # Wypisz gradienty
            
            gradients = model.backward(inputs, target)

            # Aktualizacja wag
            model.update_weights(learning_rate, *gradients)

            total_reward += reward

    #print(model.weights_output)
    #print(reward)
    #print(total_reward)
    print(death_count)
    #print(action)







    ####

    font = pygame.font.Font('freesansbold.ttf', 20)

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed +=1

        text = font.render("Points: " + str(points), True, (0,0,0))
        textRect = text.get_rect()
        textRect.center = (1000,40)
        screen.blit(text, textRect)

    def backgroundDraw():
        global x_pos_bg, y_pos_bg

        image_width = background.get_width()
        screen.blit(background, (x_pos_bg, y_pos_bg))
        screen.blit(background, (image_width + x_pos_bg, y_pos_bg))

        if x_pos_bg <= -image_width:
            screen.blit(background, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        screen.fill((255,255,255))
        userInput = pygame.key.get_pressed()
        
        player.draw(screen)
        player.update(userInput)
        
        ####
        inputs = np.array([[obstacle_distance, obstacle_height, obstacle_width, game_speed]])
        
        
        normalized_inputs = np.array([(obstacle_distance - 300)/(150-300),
                         (obstacle_height - 65)/(97-65),
                         (obstacle_width-40)/(105-40),
                         (game_speed - 14)/(100-14)])

        target_options = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1],[1,0,0]])

        model.forward(inputs)
        action = np.argmax(model.output)
        reward = reward_system(points, death_count, action)
        
         

        
        #print(action)
        #print(np.argmax(model.output))

        if action == 0:
            player.dino_duck = False
            player.dino_run = True
            player.dino_jump = False
            player.run
            
        elif action == 1:
                player.dino_duck = True
                player.dino_run = False
                player.dino_jump = False
                player.duck
        elif action == 2:
                player.dino_duck = False
                player.dino_run = False
                player.dino_jump = True
                player.jump

        ####

        if len(obstacles) == 0:
            if random.randint(0,2) == 0:
                obstacles.append(SmallCactus(smallCactus))
            elif random.randint(0,2) == 1:
                obstacles.append(LargeCactus(largeCactus))
            elif random.randint(0,2) == 2:
                obstacles.append(Bird(bird))
        

        for obstacle in obstacles:
            obstacle.draw(screen)
            obstacle.update()
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(0)
                death_count += 1
                menu(death_count)

        try:
            obstacle_distance = obstacles[-1].rect.x
            obstacle_width = obstacles[-1].rect.width
            obstacle_height = obstacles[-1].rect.height
        except:
            pass

        backgroundDraw()
        cloud.draw(screen)
        cloud.update()
        score()        
        clock.tick(30)
        pygame.display.update()

        def menu(death_count):
            global points
            run = True
            while run:
                screen.fill((255,255,255))
                font = pygame.font.Font('freesansbold.ttf', 20)

                if death_count == 0:
                    text = font.render("Press any key to start", True, (0,0,0))
                elif death_count > 0:
                    text = font.render("Press any key to restart", True, (0,0,0))
                    score = font.render("Your score: " +str(points), True, (0,0,0))
                    scoreRect = score.get_rect()
                    scoreRect.center = (screenW //2, screenH // 2 + 50)
                    screen.blit(score, scoreRect)
                textRect = text.get_rect()
                textRect.center = (screenW // 2, screenH // 2)
                screen.blit(text, textRect)
                pygame.display.update()
                main()
                for event in pygame.event.get():
                    if event.type ==pygame.QUIT:
                        run = False
                    #if event.type == pygame.KEYDOWN:
                        #main()
                    else:
                        main()