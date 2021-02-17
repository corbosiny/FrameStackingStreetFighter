previous_player1_health = 175
previous_player1_matches_won = 0

previous_player2_health = 175
previous_player2_matches_won = 0

function calculate_reward()
  reward = 0
  
  if data.player1_health < previous_player1_health then
    local delta = data.player1_health - previous_player1_health
    previous_player1_health = data.player1_health
    reward = reward + delta
  end
  
  if data.player2_health < previous_player2_health then
    local delta = previous_player2_health - data.player2_health
    previous_player2_health = data.player2_health
    reward = reward + delta
  end
  
  if data.player1_matches_won > previous_player1_matches_won then
    local delta = 100
    previous_player1_matches_won = data.matches_won
    reward = reward + delta
  end
  
  if data.player2_matches_won > previous_player2_matches_won then
    local delta = -100
    previous_player2_matches_won = data.enemy_matches_won
    reward = reward + delta
  end
  
  return reward
  
end
