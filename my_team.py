##########
# Import #
##########

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#################
# Team creation #
#################

# verander niet

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


###############
# Base Agent #
###############

# deze is de basis agent die we zullen gebruiken om de aanvaller en verdediger te implementeren
# => niet veranderen (???)

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


###################
# Offensive Agent #
###################

# deze agent gaat aanvallen

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def __init__(self, index):
        super().__init__(index)
        self.food_collected = 0  # Houdt bij hoeveel voedsel is verzameld: handig om hem te laten spelen
        self.previous_score = 0 # We houden dit bij om te zien als score verhoogd werd of niet
        self.return_to_base = False # hij moet terug naar base als de flag op true staat
        self.enemy_scared = False # als enemy ghost scared zijn willen we zoveel mogelijk eten

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # als hij voedsel opeet => teller moet naar omhoog
        #if my_pos in food_list:
         #   self.food_collected += 1

        return features

    def choose_action(self, game_state):

        """ 0. Variabelen die we overal nodig zullen hebben """
        actions = game_state.get_legal_actions(self.index) # capture.py: lijn 108
        current_position = game_state.get_agent_position(self.index) # capture.py: lijn 131
        current_score = self.get_score(game_state) # capture.py: lijn 145 <-- na replay: deze kan ook negatief zijn

        """ 1. Identificeer vijanden + ontsnappen aan vijand (als hij niet bang is) --> enkel ghost ontsnappen """
        # => list comphrension gebruikt: https://www.w3schools.com/python/python_lists_comprehension.asp

        # Variabelen die we hier nodig zullen hebben
        enemy_indices = self.get_opponents(game_state) # capture_agents.py: lijn 221 => identificatie van onze vijanden
        dangerous_enemies = []  # dangerous enemies zijn enkel ghost enemies en niet pacman enemies
        dangerous_enemy_distance_treshold = 5 # als hij binnen 5 stappen dichtbij is => ontsnappen
        # enemy_positions = [game_state.get_agent_position(index) for index in enemy_indices] # capture.py: lijn 131 => positie van onze vijanden

        # 1.1: Loop door alle enemies, en zoek normale ghosts (die zijn dangerous)
        for index in enemy_indices:
            enemy_state = game_state.get_agent_state(index) # capture.py: lijn 128 => kan spook of pacman zijn

            # Check of de vijand een geen pacman is en niet scared is
            if enemy_state.is_pacman == False: # ik heb dat zien gebruiken bij lijn 486 in capture.py

                if enemy_state.scared_timer == 0: # deze enemy mag niet scared zijn
                    enemy_position = game_state.get_agent_position(index) # haal positie op van die spook
                    self.enemy_scared = False # timer staat op 0: je moet nu wel wegrennen van ghost
                #    print(f"- scared: {self.enemy_scared}")

                    # Voeg de vijand toe als het binnen de gevarenafstand valt
                    if enemy_position and self.get_maze_distance(enemy_position, current_position) < dangerous_enemy_distance_treshold: # capture_agents.py: lijn 252 => distance between two points: gaat "position" bevatten
                        dangerous_enemies.append(enemy_position)

                else: # hij is scared
                    self.enemy_scared = True
                   # print(f"- scared: {self.enemy_scared}")

        # 1.2: Ontwijk deze dangerous ghosts --> dit moet beter gemaakt worden. "1.1" moet je niet aanraken normaal gezien
        if dangerous_enemies: # zijn er dangerous enemies?

            # 1.2.1: Kies de actie die het verst van de vijand wegleidt: die is veilig
            safe_actions = [] # hierin gaan alle veilige acties zitten
            max_safe_distance = 0 # in het begin is dit gewoon 0, maar we gaan dat wel veranderen.

            for action in actions:
                successor = self.get_successor(game_state, action)
                next_position = successor.get_agent_position(self.index)

                # De minimale afstand berekenen tot de vijand voor de huidige actie
                min_distance_to_enemy = min([self.get_maze_distance(next_position, dangerous_enemy) for dangerous_enemy in dangerous_enemies])

                # We kiezen voor de actie die het verst van de vijand af ligt
                if min_distance_to_enemy > max_safe_distance:
                    max_safe_distance = min_distance_to_enemy # deze wordt de nieuwe veilige afstand
                    safe_actions = [action] # Deze is de enige veilige actie
                elif min_distance_to_enemy == max_safe_distance:
                    safe_actions.append(action) # Deze is een veilige actie, die we toevoegen aan andere veilige acties

            # 1.2.2: Van die safe_actions, kies de beste safe actions en niet zomaar random kiezen
            walls = game_state.get_walls()  # capture.py: lijn 171 => we willen acties die niet naar een muur brengt

            # We willen enkel de juiste directions kiezen voor deze actie
            action_to_direction = {
                # De verschillende directions bekijken in de volgende positie: Noord, Zuid, Oost, West -> met print gechecked
                "North": (0, 1),
                "South": (0, -1),
                "East": (1, 0),
                "West": (-1, 0),
            }

            safe_directions = [action_to_direction[safe_action] for safe_action in safe_actions if safe_action in action_to_direction]

            if safe_actions:
                best_safe_action = None
                lookahead_steps = 10 # We willen zoveel stappen verder kijken of we niet in een dode hoek belanden
                best_future_steps = 0  # Hoeveel stappen kan Pacman veilig bewegen?

                # We controleren of de acties Pacman niet in een hoek brengen
                # Bijvoorbeeld: als hij south gaat dan zijn er walls links, rechts, en beneden -> dan moet hij daarin niet gaan, hij moet een andere actie kiezen

                # Check alle veilige acties
                for safe_action in safe_actions:
                    successor = self.get_successor(game_state, safe_action)
                    next_position = successor.get_agent_position(self.index)

                    # Voor elke direction gaan we kijken hoe ver we kunnen gaan
                    for direction in safe_directions:
                        dx, dy = direction # extraheer de x, y coördinaten (wiskunde notatie genomen om aan te tonen dat we zoveel verder gaan)
                        next_x, next_y = next_position  # https://www.geeksforgeeks.org/python-assign-multiple-variables-with-list-values/
                        steps_taken = 0  # Tel hoeveel stappen we genomen hebben

                        # Kijk hoe ver Pacman kan gaan voordat hij een muur raakt met deze direction
                        while steps_taken < lookahead_steps and not walls[next_x][next_y]:
                            steps_taken += 1
                            next_x, next_y = next_x + dx, next_y + dy # Bereken de toekomstige positie

                        # Als we een pad vinden dat langer is dan de beste eerdere pad -> dan is deze de nieuwe beste pad
                        if steps_taken > best_future_steps:
                            best_safe_action = safe_action
                            best_future_steps = steps_taken

                        # Als we het maximale aantal stappen hebben bereikt, stoppen we met deze richting
                        if steps_taken >= lookahead_steps:
                            break  # We hebben een lange veilige pad gevonden, dus stoppen we de innerlijke loop

                    # We zitten buiten de binnenste for-loop: check of we de volgende safe_action moeten checken of niet
                    if best_future_steps >= lookahead_steps:
                        break # We hebben een beste pad gevonden

                # Na het controleren van alle richtingen voor deze actie, kies de beste actie
                if best_safe_action:
                  #  print(f"- Beste safe action: {best_safe_action}")
                    return best_safe_action



                    # while steps_taken < lookahead_steps:
                    #     possible_moves = [] # Lijst van mogelijke posities waar Pacman naartoe kan bewegen (waar geen wall is)
                    #
                    #     for direction in directions:
                    #         x, y = direction  # haal de x en y eruit
                    #         check_x, check_y = future_x + x, future_y + y # die verhogen we met de next om te checken
                    #
                    #         # Controleer of er geen muren zijn in de volgende positie
                    #         if not walls[check_x][check_y]:
                    #             possible_moves.append((check_x, check_y)) # We voegen deze positie toe: we mogen daar gaan
                    #
                    #     # We checken nu of deze actie goed is
                    #     if len(possible_moves) == 0: # Betekent dat er overal muren waren
                    #         future_safe = False # We zeggen dat deze route niet goed is
                    #         break # We stoppen met de while loop, zodat we een volgende safe actie kunnen checken en niet zomaar met deze actie verder blijven kijken
                    #     else: # Voor alle andere gevallen: Anders is het goed, we gaan naar die positie bewegen
                    #         future_x, future_y = possible_moves[0] # We bewegen naar de eerst mogelijke plek --> in de volgende iteratie gaan we met deze pad verder gaan
                    #         steps_taken += 1 # We hebben een stap bekeken
                    #
                    # # Als deze pad safe is: we gaan deze actie gewoon gebruiken
                    # if future_safe == True and len(possible_moves) > best_future_steps:
                    #     best_future_steps = len(possible_moves)
                    #     best_safe_action = safe_action # deze actie is het meest safe (voorlopig, tot we niet uit de for-loop geraken)

                # # Als er een beste safe action is, geef deze terug
                # if best_safe_action:
                #     print(f"Best Safe Action: {best_safe_action}")
                #     return best_safe_action
                # else:
                #     return random.choice(safe_actions) # anders kiezen we random uit de safe_actions: zodat we daarmee verder kunnen (de enemy defender gaat ook spelen, dus kan in volgende iteratie andere safe_actions zijn)
                #     print("Geen beste safe action gevonden")

                # ___________________

                    # for direction in directions:
                    #     x, y = direction # haal de x en y eruit
                    #     check_x, check_y = next_x + x, next_y + y # die verhogen we met de next om te checken
                    #
                    #     # Controleer of er geen muren zijn in de volgende positie
                    #     if not walls[check_x][check_y]:
                    #         free_spaces += 1

                    # # Nu gaan we de actie kieze met meeste "vrije ruimte" rondom de volgende positie
                    # if free_spaces > best_free_spaces:
                    #     best_free_spaces = free_spaces
                    #     best_safe_action = safe_action

                    # Ook enkele posities verder checken of we niet vast zullen zitten met deze huidige actie

                # # Als er een beste safe action is, geef deze terug
                # if best_safe_action:
                #     return best_safe_action
                # else:
                #     print("Geen beste safe action gevonden")

            #     return random.choice(safe_actions)  # Kies een willekeurige veilige actie
            # else:
            #     return random.choice(actions) # Fallback: hij beweegde soms niet meer

        """ 2. Dichtstbijzijnde capsule opeten indien zijn afstand kleiner is dan treshold """

        # Variabelen die we hier nodig zullen hebben
        capsules = self.get_capsules(game_state) # capture.py => lijn 220: haal alle capsules op
        capsule_distance_treshold = 4 # als de capsule twee stappen verder ligt: eet hem op

        # we moeten ook eerst zien of er capsules zijn
        if capsules:
            closest_capsule = None
            min_capsule_distance = float('inf')

            # 2.1: We zoeken dichtstbijzijnde capsule die binnen de treshold ligt
            for capsule in capsules:
                distance = self.get_maze_distance(current_position, capsule)
                is_it_close = distance <= capsule_distance_treshold # hij moet dichtbij zijn
                if distance < min_capsule_distance and is_it_close == True:
                    closest_capsule = capsule
                    min_capsule_distance = distance

            # 2.2: Nu willen we hem zo dicht mogelijk bij deze capsule brengen door de beste actie te kiezen
            if closest_capsule is not None:
                best_action = None
                best_distance = float('inf')

                for action in actions:
                    successor = self.get_successor(game_state, action)
                    next_position = successor.get_agent_position(self.index)
                    distance = self.get_maze_distance(next_position, closest_capsule)
                    if distance < best_distance:
                        best_action = action
                        best_distance = distance

                # Retourneer beste actie
                if best_action:
                    return best_action

        """ 3. Dichtstbijzijnde voedselbron zoeken door afstanden te vergelijken """

        # Variabelen die we hier nodig zullen hebben
        food_collected_treshold = 2 # (er zijn 20 voedselbronnen) => als hij meer dan 2 voedselbron heeft opgegeten willen we hem terug naar base brengen
        food_list = self.get_food(game_state).as_list() # capture_agents.py: lijn 188

        # Het kan zijn dat je opgegeten werd wanneer je voedselbronnen aan het opeten was: een bug dat ik zag (hij ging minder dan de treshold opeten)
        if current_position == self.start and self.food_collected > 0:
          #  print("hij werd opgegeten, wanneer hij voedsel aan het opeten was")
            self.food_collected = 0  # We resetten dit, zodat we weer goed beginnen tellen

        # We willen enkel blijven eten als hij minder dan de treshold heeft opgegeten, en dat hij nog niet naar de base moet gaan
        if food_list and self.return_to_base == False:

            # 3.1: We gaan eerst de dichtstbijzijnde voedsel bron zoeken
            closest_food = None # de voedselbron die het dichtbijzijnste is
            best_distance = float('inf')

            for food in food_list:
                distance = self.get_maze_distance(current_position, food)
                if distance < best_distance:
                 closest_food = food
                 best_distance = distance

            # 3.2: Nu willen we hem naar de dichtstbijzijnde food brengen, door de juiste actie te kiezen
            if closest_food is not None:
                best_action = None
                best_distance = float('inf') # we gaan deze variabele hergebruiken

                # Loop door alle mogelijke acties, om de beste actie te vinden
                for action in actions:
                    successor = self.get_successor(game_state, action) # kijk waar de agent terechtkomt na deze actie
                    next_position = successor.get_agent_position(self.index) # krijg de nieuwe positie van de agent in de successor state
                    distance = self.get_maze_distance(next_position, closest_food) # de distance van de volgende positie naar de dichtstbijzijnde voedselbron
                    if distance < best_distance: # als deze actie dichterbij de voedselbron brengt, wordt het voorlopig de beste actie (we moeten nog verder loopen)
                        best_action = action
                        best_distance = distance

                # Controleer of de agent in de successor state met de beste actie op dezelfde positie als de closest_food is => verhoog counter
                successor = self.get_successor(game_state, best_action)
                next_position = successor.get_agent_position(self.index)

                if next_position == closest_food:
                    self.food_collected += 1
                   # print(f"- gegeten: {self.food_collected}")

                # Return to base flag veranderen indien we genoeg gegeten hebben
                if self.food_collected < food_collected_treshold: # nog niet genoeg gegeten
                    self.return_to_base = False
                    #print("- niet genoeg gegeten")

                elif self.food_collected == food_collected_treshold: # genoeg gegeten
                    self.return_to_base = not self.enemy_scared # als enemy_scared op "True" staat gaat dankzij de "not" het geflipt worden: dus wordt het "False" => we blijven verder eten
                    #print(f"- genoeg gegeten, terug naar base: {self.return_to_base}")

                elif self.food_collected > food_collected_treshold: # We hebben te veel gegeten omdat enemy bang was
                    if self.enemy_scared == True: # is hij nog altijd bang?
                        self.return_to_base = False # blijf verder eten
                       # print(f"- meer gegeten, enemy scared -> niet naar base")
                    else: # hij is niet meer bang
                        self.return_to_base = True # ga terug naar base
                      #  print(f"- meer gegeten, enemy not scared -> naar base")

                # if self.food_collected >= food_collected_treshold:
                #     # Hij mag ook enkel naar base wanneer de ghost niet scared is. Is die scared, eet zoveel mogelijk op
                #     if self.enemy_scared == False:
                #        self.return_to_base = True
                #     if self.enemy_scared == True:
                #         self.return_to_base = False

                # Retourneer beste actie
                if best_action:
                    return best_action
        else: # anders hebben we alle voedselbronnen opgegeten -> we moeten hem expliciet zeggen ga terug naar base
            self.return_to_base = True

        """ 4. Terug naar base gaan om score te verhogen """

        # Enkel terug naar base gaan als flag op True staat
        if self.return_to_base == True:
            best_distance = float('inf')
            best_action = None

            # We berekenen afstand tot de start positie: maar dit willen we eigenlijk niet
            for action in actions:
                successor = self.get_successor(game_state, action)
                next_position = successor.get_agent_position(self.index)
                distance_to_start_position = self.get_maze_distance(self.start, next_position)
                if distance_to_start_position < best_distance:
                    best_action = action
                    best_distance = distance_to_start_position

            # We willen niet dat onze agent tot de start positie gaat: we passen een trucje
            # --> als current_score > previous score => betekent dat hij zijn kant bereikt heeft
            # oude (na replay verbeterd): if current_score > self.previous_score or current_position == self.start:
            #      --> score kon ook negatief zijn: daardoor ging hij volledig naar de start positie
            #      --> fix: beide situaties checken: ">" en "<"

            # Score was negatief tot we base geraakt zijn + kan ook kleiner worden doordat opponent voedsel op gegeten heeft
            if current_score < self.previous_score:
                self.previous_score = current_score  # we updaten de previous score, zodat we goed tellen voor verdere berekeningen
                self.food_collected = 0  # Reset de teller: zodat hij weer begint te aanvallen
                self.return_to_base = True # Forceer hem om nog altijd naar de base te gaan

            # Als we tot de base geraakt zijn: score gaat verhoogd worden, dus ">"
            elif current_score > self.previous_score:
                self.previous_score = current_score  # we updaten de previous score, zodat we goed tellen voor verdere berekeningen
                self.food_collected = 0  # Reset de teller: zodat hij weer begint te aanvallen
                self.return_to_base = False  # Hij mag nu weer beginnen eten

            # Het kan zijn dat je opgegeten werd wanneer je naar base wou gaan
            elif current_position == self.start:
                self.previous_score = current_score
                self.food_collected = 0
                self.return_to_base = False

            return best_action  # kies de actie die de kortste afstand naar de base heeft

        """ 5. Anders kies een actie op basis van de evaluatie van de situatie
               Als ik deze niet deed kreeg ik error -> Want gaf gewoon NONE terug (Exception: Illegal action None) """
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions) # we gaan random eruit kiezen

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

###################
# Defensive Agent #
###################

# deze agent gaat verdedigen

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free.
    This version focuses on defending while avoiding unnecessary movements.
    When scared, it switches to offensive behavior.
    """
    '''hier gaan we variabelen en flags definiëren, die we later gaan gebruiken'''

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        # Adding attributes needed for offensive behavior when scared
        self.food_collected = 0
        self.previous_score = 0
        self.return_to_base = False
        self.enemy_scared = False
        self.scared_timer_off = True  # Flag to track if the scared timer is done
        # Offensive agent will be initialized in register_initial_state
        self.offensive_agent = None
        # Maak de offensive agent aan
        self.offensive_agent = OffensiveReflexAgent(self.index)

        # Verwijs naar zichzelf als defensive agent
        self.defensive_agent = self


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        # Initialize the offensive agent with the same index
        self.offensive_agent = OffensiveReflexAgent(self.index)
        self.offensive_agent.register_initial_state(game_state)

    '''hier definieren we een test die checkt of een agent scared is'''

    def is_scared(self, game_state):
        """Check if this agent is in a scared state"""
        my_state = game_state.get_agent_state(self.index)
        return my_state.scared_timer > 0



    def choose_action(self, game_state):
        # Get current state information
        my_state = game_state.get_agent_state(self.index)
        is_scared = my_state.scared_timer > 0

        # Check for invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # If there are invaders, prioritize defense regardless of scared status
        if invaders:
            print("- Switch to defense")
            actions = game_state.get_legal_actions(self.index)
            best_dist = float('inf')
            best_action = None

            for action in actions:
                successor = self.get_successor(game_state, action)
                pos = successor.get_agent_position(self.index)

                if pos is not None:
                    dists = [self.get_maze_distance(pos, inv.get_position()) for inv in invaders
                             if inv.get_position() is not None]
                    if dists:
                        dist = min(dists)
                        if dist < best_dist:
                            best_dist = dist
                            best_action = action

            if best_action:
                return best_action

        # If we're scared or no invaders, switch to offensive behavior
        elif not invaders or is_scared:
            if not self.return_to_base: # hij moet nog niet naar base terug komen
                print(f"- Switch to offense")
                # Update offensive agent's state variables
                self.offensive_agent.food_collected = self.food_collected
                self.offensive_agent.previous_score = self.previous_score
                self.offensive_agent.return_to_base = self.return_to_base
                self.offensive_agent.start = self.start

                # Use offensive agent's choose_action method
                action = self.offensive_agent.choose_action(game_state)

                # Update our state variables with any changes made
                self.food_collected = self.offensive_agent.food_collected
                self.previous_score = self.offensive_agent.previous_score
                self.return_to_base = self.offensive_agent.return_to_base

                return action

        # Als ik een pacman ben, en er zijn invaders in mijn base / ik ben niet meer scared -> ga direct terug
        elif (my_state.is_pacman and invaders) or not is_scared:
            print("- Back to base")
            # Use defensive evaluation to get back to our side
            actions = game_state.get_legal_actions(self.index)
            best_score = float('-inf')
            best_action = None

            for action in actions:
                successor = self.get_successor(game_state, action)
                next_state = successor.get_agent_state(self.index)
                next_pos = next_state.get_position()

                # Calculate score based on distance to our base
                if next_pos is not None and self.start is not None:
                    score = -self.get_maze_distance(next_pos,
                                                    self.start)  # Negative because we want to minimize distance

                    # Prefer actions that make us not pacman (back on our side)
                    if not next_state.is_pacman:
                        score += 100

                    if score > best_score:
                        best_score = score
                        best_action = action

            if best_action:
                return best_action

        # Default defensive behavior
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        is_scared = my_state.scared_timer > 0
        my_pos = my_state.get_position()

        # Check for invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if my_pos is None:
            return features

        # features['on_defense'] = 1 # In het begin defend je altijd
        #
        # if invaders:
        #     features['on_defense'] = 1 # defend
        # elif not invaders or is_scared:
        #     features['on_defense'] = 0 # niet defenden
        # elif (my_state.is_pacman and invaders) or not is_scared:
        #     features['on_defense'] = 1 # defend

        # if my_state.is_pacman or is_scared: # ben ik een pacman of ben ik scared?
        #     if not invaders or is_scared: # Zijn er geen invaders in mijn kant? Of ben ik zelf bang?
        #         features['on_defense'] = 0 # defend niet
        #     else:
        #         features['on_defense'] = 1 # defend

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = []
        invader_positions = []

        for a in enemies:
            if a.is_pacman and a.get_position() is not None:
                invaders.append(a)
                invader_positions.append(a.get_position())

        if invader_positions:
            features['invader_distance'] = min(
                [self.get_maze_distance(my_pos, pos) for pos in invader_positions]
            )

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        if self.start is not None:
            distance_from_start = self.get_maze_distance(my_pos, self.start)
            if distance_from_start < 15:
                features['at_start'] = 1

        return features

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        print("Action:", action, "Features:", features, "Score:", features * weights)

        return features * weights

    def get_weights(self, game_state, action):
        return {
            'invader_distance': 6000,
            'on_defense': 1000,
            'stop': -100,
            'reverse': -2,
            'at_start': -15
        }