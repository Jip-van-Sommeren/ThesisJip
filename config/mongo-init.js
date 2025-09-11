// MongoDB Initialization Script for Agent System

// Switch to agent_system database
db = db.getSiblingDB('agent_system');

// Create agent_user with read/write access
db.createUser({
  user: 'agent_user',
  pwd: 'password123',
  roles: [
    { role: 'readWrite', db: 'agent_system' }
  ]
});

// Create indexes for better performance
db.agent_profiles.createIndex({ "agent_id": 1 }, { unique: true });
db.agent_profiles.createIndex({ "agent_type": 1 });
db.agent_profiles.createIndex({ "created_at": -1 });

db.role_definitions.createIndex({ "name": 1 }, { unique: true });
db.group_configurations.createIndex({ "name": 1 }, { unique: true });

// Create collections with validation
db.createCollection("agent_profiles", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["agent_id", "agent_type"],
      properties: {
        agent_id: {
          bsonType: "string",
          description: "Unique agent identifier - required"
        },
        agent_type: {
          bsonType: "string",
          enum: ["reactive", "bdi", "hybrid"],
          description: "Agent type - required"
        },
        roles: {
          bsonType: "array",
          items: {
            bsonType: "string"
          },
          description: "Agent roles"
        },
        created_at: {
          bsonType: "date",
          description: "Creation timestamp"
        }
      }
    }
  }
});

db.createCollection("role_definitions", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "description"],
      properties: {
        name: {
          bsonType: "string",
          description: "Role name - required"
        },
        description: {
          bsonType: "string",
          description: "Role description - required"
        },
        responsibilities: {
          bsonType: "array",
          description: "Role responsibilities"
        },
        permissions: {
          bsonType: "array",
          description: "Role permissions"
        }
      }
    }
  }
});

db.createCollection("group_configurations", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "purpose"],
      properties: {
        name: {
          bsonType: "string",
          description: "Group name - required"
        },
        purpose: {
          bsonType: "string",
          description: "Group purpose - required"
        },
        roles: {
          bsonType: "array",
          items: {
            bsonType: "string"
          },
          description: "Required roles in group"
        },
        agents: {
          bsonType: "array",
          items: {
            bsonType: "string"
          },
          description: "Agents in group"
        }
      }
    }
  }
});

print('Agent system database initialized successfully');