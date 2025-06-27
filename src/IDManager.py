import logging 

logger = logging.getLogger(__name__)

class GlobalIdentityManager:
    """
    Manages the mapping from view-specific track IDs to a global ID.
    """

    def __init__(self) -> None:
        self.next_global_id :int = 1000
        self.id_map = dict()
        """
        id_map = { (view_name, track_id) : global_id}
        """
        self.reverse_map = dict()
        """
        reverse_map = {global_id : [(view_name, track_id)]}
        """

        

    def get_global_id(self, view_name :str , track_id : int) -> int:
        """
        Gets the global ID for a view-specific track. If not found, creates a new one.
        """
        if(view_name, track_id) in self.id_map:
            return self.id_map[(view_name, track_id)]
        else:
            new_id = self.next_global_id
            self.id_map[(view_name, track_id)] = new_id
            self.reverse_map[new_id] = [(view_name, track_id)]
            self.get_global_id += 1
            return new_id
        

    def register(self, player1_data, player2_data) -> int:
        """
        Registers a match and assigns a consistent global ID.
        """
        key_1 = (player1_data['view'], player1_data['track_id'])
        key_2 = (player2_data['view'], player2_data['track_id'])

        global_id_1 = self.id_map.get(key_1)
        global_id_2 = self.id_map.get(key_2)

        if global_id_1 is not None and global_id_2 is not None:
            "This case shouldn't happen often with good tracking."
            return global_id_1
        elif global_id_1 is not None:
            self.id_map[key_2] = global_id_1
            return global_id_1
        elif global_id_2 is not None:
            self.id_map[key_1] = global_id_2
            return global_id_2
        else:
            new_id = self.next_global_id
            self.id_map[key_1] = new_id
            self.id_map[key_2] = new_id
            self.reverse_map[new_id] = [key_1, key_2]
            self.next_global_id += 1
            logger.info(f"Assigned new Global_ID: {new_id} to Player 1:{key_1[1]} and Player 2:{key_2[1]}")
            return new_id

        